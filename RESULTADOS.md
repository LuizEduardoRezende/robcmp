# Resultados e achados — material para a dissertação

Registro dos achados que surgem durante o desenvolvimento e **para qual seção do
projeto de pesquisa cada um serve**. A numeração das seções é a do `CLAUDE.md`.

Cada entrada traz a evidência (comando, arquivo:linha, número medido) para que
possa ser reproduzida e citada na escrita sem depender de memória.

Este arquivo não é a dissertação nem substitui o projeto de pesquisa — é a ponte
entre o que acontece no código e o que vai ser escrito.

---

## Seção 4 — Caracterização do baseline do TCC

### 4.1 O wrapper do TCC compila sem alteração contra o TFLM atual

Importado de `tcc/tflm-front-end` e compilado contra o TFLM fixado em
`d0318206` (main de 2026-09-11), **277 commits à frente** do pin que o TCC usava
(`8eb17a7a`, 2025-08-22). Zero erros, zero warnings, incluindo com os 99 kernels
que o `switch` cobre todos habilitados.

Dos dois headers da API que mudaram no intervalo, nenhuma mudança atinge o
wrapper: são aditivas (4 métodos novos) e correção de typo em membro privado. A
única alteração de assinatura real (`SetDecompressionMemory`) não é usada.

**Serve para:** estabelecer que o baseline é comparável sem adaptação, o que
valida a comparação retroativa exigida na 5.4.

### 4.2 Duas funções da API nunca foram implementadas

`AnalyzeModelKernels` e `RunBenchmarkOptimized` estavam declaradas em
`wrappers/tflm/tflm_wrapper.h` e **sem definição** no `.cpp`. Compilam sem
reclamar; qualquer `.rob` que as chamasse falharia na linkedição.

**Serve para:** a 5.2. O `AnalyzeModelKernels` seria o embrião da análise de
kernels em runtime. Como nunca existiu de fato, o módulo de análise estática
**não substitui código funcionando — implementa do zero**. A comparação com o
baseline precisa refletir isso.

### 4.3 A descrição do Micro Speech no projeto precisa de ajuste

O projeto descreve "espectrograma de 49 features". Medido com `tflite_info`:
a entrada do `micro_speech_quantized.tflite` é `int8[1,1960]`, e
1960 = 49 × 40 — são 49 janelas temporais de 40 canais de filterbank. O
`audio_preprocessor` produz 40 valores por chamada, ou seja, **49 invocações do
preprocessor alimentam 1 do classificador**.

**Serve para:** corrigir a descrição na 4.4 e para dimensionar o experimento de
pipeline na 5.4.

---

## Seção 5.1 — Refatoração do wrapper

### 5.1.1 A configuração manual de kernels é uma cópia que apodrece

O `kernel_config.h` duplica à mão informação que o header do TFLM já carrega.
Auditoria contra `micro_mutable_op_resolver.h` no pin atual:

- 121 métodos `Add*` disponíveis no TFLM
- 99 com `case` no `switch` do `RegisterOp`
- **3 anotações "não suportado" incorretas** — `REDUCE_ALL` e
  `DYNAMIC_UPDATE_SLICE` passaram a existir no upstream, e `CUSTOM` nunca foi
  kernel (é o mecanismo genérico de registro por string)
- **19 kernels disponíveis e ausentes do enum**, porque são custom ops
  registrados por string e não entradas de `BuiltinOperator`
- 1 `#define` duplicado
- 1 `case` chamando `AddRandomStandardNormal()`, símbolo que **nunca existiu no
  upstream**, inalcançável por configuração

Duas das três anotações erradas nasceram nos 277 commits entre os dois pins.

**Serve para:** justificar empiricamente a eliminação do `kernel_config.h`. Não
é preferência de estilo — é um artefato que diverge a cada atualização do
submódulo.

### 5.1.2 Por que o switch monolítico impede a eliminação por linkedição

O linker resolve na granularidade do símbolo. Com `RegisterOp` sendo **uma
função** contendo as 99 chamadas, um `programa.o` com `U RegisterOp` faz o
linker puxar a função inteira, e com ela as referências aos 99 `Register_*` do
TFLM. Os `#if ENABLE_*` existem para contornar isso no pré-processador.

Verificado com `nm` sobre objeto gerado pelo robcmp: chamadas a funções sem
corpo aparecem como símbolos `U`, resolvidos contra o `.a` na linkedição.

**Serve para:** o argumento técnico central da 5.1. Separar em funções por
kernel torna cada uma um símbolo independente, e a eliminação passa a ser
consequência da granularidade, não de configuração manual.

### 5.1.3 O wrapper usava heap (resolvido em B5)

Eram 13 ocorrências de `new`/`delete` e um `malloc` em
`AllocAndGetTensorArray`. O TFLM inteiro é projetado para não alocar
dinamicamente, e a arena já chega pronta por parâmetro.

Resolvido: instâncias passam a viver em armazenamento estático
(`g_instances[TFLM_MAX_MODELS]`), o `MicroInterpreter` é construído com
*placement new* sobre buffer alinhado, e o `MicroAllocator` já alocava dentro da
própria arena. `AllocAndGetTensorArray` foi removida — usava `malloc` e nenhum
`.rob` a chamava.

### 5.1.4 A eliminação por linkedição funciona — medida

Experimento controlado, antes de aplicar ao wrapper: três funções independentes
num **único** `.cpp`, compiladas com `-ffunction-sections`, arquivadas em `.a`,
e um programa que chama apenas uma.

```
sem --gc-sections:  tflm_add_conv_2d  tflm_add_softmax  tflm_add_avg_pool
com --gc-sections:  tflm_add_conv_2d
```

Conclusão de layout: **não é preciso um arquivo por kernel**. A granularidade de
seção basta, e as flags já estavam em uso no projeto — o Makefile do TCC já
tinha `-ffunction-sections -fdata-sections` e `-Wl,--gc-sections`, e
`platformio/platform/builder/main.py:58` acrescenta `--gc-sections` a toda
linkedição.

Aplicado ao wrapper real, um programa que chama `tflm_add_conv_2d` fica com
**1 das 98** funções no binário, e o despachante inteiro é descartado junto.

### 5.1.5 Dois bugs de assinatura na convenção de chamada

A convenção do Robcmp, verificada emitindo LLVM IR:

```
void f(uint64 h)                     -> declare void @f(i64)
void g(uint64 h, uint8[] d)          -> declare void @g(i64, ptr, i32)
void h(uint8[] a, uint8[] b, uint8[] c) -> declare void @h(ptr, ptr, ptr, i32, i32, i32)
```

Um `int` extra **por vetor**, todos apendados ao fim, como `i32`. A partir disso,
dois defeitos no wrapper do TCC:

1. **Parâmetro fantasma.** `GetTensorSize`, `GetInputTensor`, `SetTensorValue`,
   `InvokeInterpreter` e outras declaram um `int` final sem ter argumento vetor
   algum. A RL nunca passa esse valor. Funciona por acidente da ABI.
2. **Largura errada.** `InitializeInterpreter` declarava o terceiro tamanho como
   `int8_t num_kernels`, enquanto a RL passa `i32`. Lê 8 bits de um valor de 32;
   funciona em little-endian por sorte.

**Serve para:** justificar a revisão das assinaturas na refatoração, e como
exemplo concreto de erro que a convenção implícita do compilador induz — material
para a discussão sobre a convenção de vetores.

### 5.1.6 O modo biblioteca também passa a eliminar kernels

Decisão de projeto: em vez de manter o `kernel_config.h` para o modo `ai.tflm`,
esse modo passa a chamar as mesmas funções por kernel. Os dois modos viram **o
mesmo mecanismo em níveis diferentes de automação** — na biblioteca o usuário
escreve as chamadas, na sintaxe nativa o compilador escreve por ele. A sintaxe
nativa é açúcar sobre o modo biblioteca, não um caminho paralelo.

Ganho adicional de legibilidade: o modo biblioteca escrevia
`const required_kernels = {9};` — número mágico. Passa a escrever
`tflm_add_fully_connected(h)`.

**Custo, a declarar na 5.4:** os 5 `.rob` de teste do TCC que passam
`required_kernels` precisam ser adaptados. O teste de retrocompatibilidade
compara contra um baseline **adaptado**, não idêntico.


### 5.1.7 API final: legado removido, nomes que espelham a sintaxe nativa

Com a compatibilidade com a API do TCC já abandonada (5.1.6), o legado foi
removido por completo:

- `SetTensorFromIntArray`: obsoleta. A escrita crua a substitui, com verificação
  de tipo e tamanho — a ausência dessas verificações é o que produziu o bug do
  person detection do TCC (5.4.18).
- O `count` explícito de `SetTensorArray`/`GetTensorArray`: o usuário escrevia
  o tamanho e a RL injetava o tamanho real ao lado. Se divergissem, a cópia saía
  do vetor. Agora vale só o tamanho injetado, que precisa ser exatamente o
  número de elementos do tensor.
- Os nomes em PascalCase do TCC, que conviviam com os `tflm_` novos.

A API resultante espelha a sintaxe nativa um para um:

| Biblioteca (`ai.tflm`) | Sintaxe nativa |
|---|---|
| `tflm_input(m, i)` / `tflm_output(m, i)` | `m.input[i]` / `m.output[i]` |
| `tflm_invoke(m)` | `m.invoke()` |
| `tflm_set_values(t, v)` / `tflm_get_values(t, v)` | `m.input[i] = v` / `v = m.output[i]` |
| `tflm_set_value(t, j, x)` / `tflm_get_value(t, j)` | `m.input[i][j] = x` / `x = m.output[i][j]` |
| `tflm_set_raw_int8(t, v)` / `tflm_get_raw_int8(t, v)` | `m.input[i].raw = v` / `v = m.output[i].raw` |
| `tflm_tensor_size(t)` | `m.input[i].size` |

**Serve para:** o argumento de que a sintaxe nativa é açúcar sobre o modo
biblioteca (5.1.6) deixa de ser só conceitual — a correspondência é literal,
função a função.

### 5.1.8 Segunda regressão do B6, corrigida: verificação de limites

A unificação da conversão de tipos perdeu a verificação de índice que o wrapper
do TCC tinha (`if (index >= tensor_size) return 0.0f`). Um índice além do tensor
escreveria na arena. Corrigido no ponto único de despacho (`tensor_write`/
`tensor_read`), onde passa a valer para todas as funções; testado em
`validate_wrapper.cpp`.

Com esta e a 5.4.17, o B6 introduziu duas regressões, ambas achadas ao portar os
testes do TCC e compará-los com a semântica original. Lição de método: unificar
código duplicado exige comparar o comportamento de CADA cópia, inclusive as
verificações de borda, não só o caminho principal.

### 5.1.9 Handles de 32 bits no Cortex-M: toda a API quebraria no hardware

A RL declara handles e índices como `uint64`. O wrapper do TCC os recebia como
`uintptr_t` e `size_t`. No x86_64 os três têm 64 bits e coincidem por acaso. No
Cortex-M, `uintptr_t` e `size_t` têm 32:

```
RL  (robcmp -a stm32f1):  declare i64 @GetInputTensor(i64, i64)
C   (Cortex-M4):          define  i32 @GetInputTensor(i32, i32)
```

Pelo AAPCS, um argumento de 64 bits ocupa um par de registradores. A RL põe o
handle em r0:r1 e o índice em r2:r3; o C lê o índice de r1 — a metade alta do
handle, sempre zero. `GetInputTensor(h, 1)` devolveria sempre o tensor 0, e o
retorno chegaria truncado. Afetava todas as funções com handle, incluindo as
119 `tflm_add_*`. Nenhum erro de compilação nem de linkedição: só resultado
errado no STM32.

Corrigido usando `uint64_t` no C. Criado `test/tflm/check_abi.py`, parte do
`make check`: gera um programa RL que chama as 137 funções da fachada e um C++
que chama as do header, compila os dois para LLVM IR em x86_64 e Cortex-M, e
compara as assinaturas. Resultado: **0 divergências de tipo nos dois alvos**.
A regra foi registrada no `CLAUDE.md`, seção 6.

**Serve para:** a Etapa 4. Sem essa correção, a primeira execução no STM32
produziria resultados errados sem nenhuma pista da causa. E como argumento
metodológico: o TCC foi validado só em x86_64, onde esse defeito é invisível.

---

## Seção 5.2 — Frontend e análise estática

### 5.2.1 A sintaxe nativa exige apenas três produções novas de gramática

Levantamento empírico, compilando cada forma contra o robcmp:

| Construção | Gramática atual |
|---|---|
| `x = obj.campo[0]`, `x = obj.campo[0][1]` | aceita |
| `obj.campo[0] = escalar`, `obj.campo[0][1] = escalar` | aceita |
| `obj.campo = {1.0}` (literal sem índice) | aceita |
| `obj.metodo()`, `x = obj.campo.size`, `x = tipo("string")` | aceita |
| **`obj.campo[0] = {1.0}`** (P1) | erro de sintaxe |
| **`x = obj.campo[0].prop`** (P2) | erro de sintaxe |

Aplicando P1 e P2 a uma cópia da gramática e rodando o Bison:

```
baseline       1 conflito shift/reduce
+ P1           1 conflito shift/reduce
+ P1 + P2      1 conflito shift/reduce
```

Com a visão `.raw` (5.2.6), entra uma terceira produção, P3 — atribuição a
propriedade depois de um índice, `obj.campo[0].prop = v`, com expressão ou
literal de array. Medida da mesma forma: continua 1 conflito, o do baseline.

**Zero conflitos novos.** O motivo é o léxico: `{ID}(\.{ID})+` faz maximal munch
(`src/Language.l:111`), então `a.b` é sempre um único `TOK_XIDENTIFIER` e o
parser nunca vê `IDENT '.' IDENT`. Um `.` depois de `]` só pode ser a regra nova.

Ambas as produções são **aditivas**: hoje as duas formas são erro de sintaxe,
logo nenhum programa que compila muda de significado.

**Serve para:** sustentar a classificação como extensão minimamente intrusiva na
taxonomia de Mernik/Heering/Sloane, com evidência quantitativa em vez de
afirmação. O trabalho é semântico, não sintático.

### 5.2.2 O ponto do compilador onde a decisão semântica já existe

O parser sempre emite `FunctionCall` (`src/Language.y:476`). É a análise
semântica que reescreve para `ConstructorCall` ou `ForcedCast`, em
`src/semantic/PropagateTypes.cpp:338`, conforme o nome seja tipo ou função.

É nesse mesmo ponto que entra a regra de precedência que preserva
retrocompatibilidade: **função definida pelo usuário vence o tipo embutido**.

**Serve para:** demonstrar que a extensão se encaixa num mecanismo existente do
compilador, não cria um paralelo.

### 5.2.3 Membros não são tokens — e por isso não colidem

`.rows` e `.cols` são símbolos injetados no valor via `addSymbol`
(`src/Matrix.cpp:12`), e `.size` via `NamedConst` (`src/Array.cpp:11`). A
resolução é **por valor**, não por nome global.

**Serve para:** a instrução da 5.2 de "remover os tokens `.input`/`.output`"
aplica-se ao desenho do TCC 1, onde eles eram tokens do léxico. No desenho novo
eles nunca chegam a ser criados, e um tipo do usuário com membros de mesmo nome
resolve os seus próprios. O item vira "não reintroduzir".

### 5.2.4 Armadilha de leitura do `.tflite`: `builtin_code` mente

Ler `oc->builtin_code()` diretamente devolve resultado **errado e silencioso**
para modelos gerados por conversores antigos, que deixam o campo zerado e
guardam o código real em `deprecated_builtin_code`.

Observado no `spam.tflite`: leitura direta reporta `ADD ADD ADD ADD`; com
`tflite::GetBuiltinCode(oc)` (de `tensorflow/lite/schema/schema_utils.h`) revela
`GATHER MEAN FULLY_CONNECTED SOFTMAX`.

**Serve para:** requisito do módulo de análise estática. É uma falha que não dá
erro — produz a lista de kernels errada e o modelo falha só em runtime, que é
exatamente o que a 5.2 quer eliminar.

### 5.2.5 Custom ops são invisíveis a um enum derivado de `BuiltinOperator`

Os 19 kernels ausentes do wrapper são registrados por `AddCustom("string", ...)`
e aparecem no `.tflite` em `OperatorCode.custom_code`, nunca em `builtin_code`.

**Serve para:** a análise estática precisa casar **os dois campos**. Um
analisador que só lê `builtin_code` não enxerga o pipeline de áudio do Micro
Speech.

### 5.2.6 Visão `.raw`: a escolha entre valor real e código é explícita

Resposta de sintaxe à lacuna 5.4.13. Cada tensor tem duas visões:
`modelo.input[0]` (valores reais, com conversão automática) e
`modelo.input[0].raw` (os códigos como o tensor os guarda, sem conversão).
Especificado em `test/tflm/native-syntax.rob`, seção 4b, com o pipeline do
micro speech escrito na sintaxe nativa.

A alternativa descartada foi o compilador escolher a visão pelo tipo do valor
atribuído. Ela cria uma armadilha: no `hello_world_int8`,
`modelo.input[0] = {1.0}` seria o valor real 1,0 (código −87), mas
`modelo.input[0] = {1}` — literal inferido como int8 — seria lido como o código
1. Uma diferença de digitação mudaria o significado sem aviso.

Com o marcador explícito, e como o compilador conhece tipo e tamanho de cada
tensor pelo `.tflite`, a `.raw` ganha verificação em tempo de compilação: tipo
do vetor diferente do tensor e tamanho diferente do número de elementos viram
erro de compilação — no wrapper, esses mesmos casos só são detectados em
runtime (devolvem −1).

**Serve para:** a discussão da 5.2 sobre mover verificações para tempo de
compilação, e como exemplo de decisão de sintaxe tomada para eliminar
ambiguidade silenciosa.

**Nota de processo:** a API crua do wrapper (`SetTensorRaw*`) foi implementada
antes desta especificação — ordem invertida. A especificação confirmou que as
funções existentes bastam. A regra "sintaxe antes da implementação" foi
registrada no `CLAUDE.md` (seção 6).

### 5.2.7 O robcmp marca `zeroext` em retornos inteiros com sinal

Achado pelo `check_abi.py`. Em toda função que devolve `int8`, a RL declara
`zeroext i8` e o C (`int8_t`, com sinal) define `signext i8`. O atributo diz ao
backend que o chamador pode supor os bits altos zerados; quando o wrapper
devolve −1, eles não estão.

Testado: `r = tflm_invoke(0)` (que devolve −1) comparado com `!= 0`, `== -1` e
`< 0` — as três comparações acertam, porque o robcmp compara em `i8` e o backend
olha só os 8 bits baixos. O risco é latente: aparece quando o valor é ampliado
e o backend dispensa uma extensão confiando no `zeroext`.

A causa é do compilador, não do wrapper: o `int8` da RL é com sinal e deveria
gerar `signext`. **Para a Etapa 3**: corrigir a escolha do atributo por
sinalização do tipo. O `check_abi.py` reporta a divergência como aviso.

---

## Seção 5.4 — Validação e métricas

### 5.4.1 Redução de 28% no objeto por mover análise para tempo de compilação

Removendo `DiagnoseModel` e `VerifyModelData` — que faziam parsing de FlatBuffers
**dentro do firmware** — do wrapper:

| | antes | depois | |
|---|---|---|---|
| objeto (`-Wall`, x86_64) | 56.664 B | 40.792 B | **−28,0%** |
| linhas no `.cpp` | 1787 | 1671 | −116 |
| warnings | 6 | 0 | |

A redução não vem do código de diagnóstico em si, e sim do que ele arrastava:
acessadores do schema FlatBuffers e as tabelas de `EnumNameBuiltinOperator`.

**Serve para:** é uma métrica de tamanho de binário **de natureza diferente** da
eliminação de kernels, e vale separar as duas na dissertação. Esta mede o efeito
de mover análise do dispositivo para o host — e é mensurável antes mesmo do
compilador existir.

Reproduzir: comparar `wrappers/tflm/tflm_wrapper.cpp` em `e1f108a` com o estado
seguinte, compilando com
`clang++ -std=c++17 -Wall -c ... -I. -Ithird-party/tflite-micro`.

### 5.4.2 Os modelos de validação: levantamento real

Medido com `test/tflm/tflite_info` sobre `test/tflm/models/`:

| Modelo | Entrada | Saída | Kernels |
|---|---|---|---|
| hello_world_float | `float32[1,1]` | `float32[1,1]` | FULLY_CONNECTED |
| hello_world_int8 | `int8[1,1]` s=0.0244801 zp=−128 | `int8[1,1]` s=0.00829096 zp=5 | FULLY_CONNECTED |
| spam | `int32[1,20]` | `float32[1,2]` | FULLY_CONNECTED, GATHER, MEAN, SOFTMAX |
| micro_speech_quantized | `int8[1,1960]` s=0.101716 | `int8[1,4]` s=0.00390625 | DEPTHWISE_CONV_2D, FULLY_CONNECTED, RESHAPE, SOFTMAX |
| person_detect | `int8[1,96,96,1]` s=0.00784314 zp=−1 | `int8[1,2]` s=0.00390625 | AVERAGE_POOL_2D, CONV_2D, DEPTHWISE_CONV_2D, RESHAPE, SOFTMAX |
| audio_preprocessor_float | `int16[1,480]` | `float32[40]` | 5 builtin + **9 custom Signal\*** |
| audio_preprocessor_int8 | `int16[1,480]` | `int8[40]` | 9 builtin + **9 custom Signal\*** |

O `person_detect.tflite` vem do próprio submódulo do TFLM, onde é distribuído
como arquivo; o TCC o consumia como array de bytes em C++.

### 5.4.3 O pipeline completo do Micro Speech não é executável hoje

O `audio_preprocessor` exige 9 custom ops (`SignalRfft`, `SignalWindow`,
`SignalFilterBank`, `SignalFilterBankLog`, `SignalFilterBankSquareRoot`,
`SignalFilterBankSpectralSubtraction`, `SignalPCAN`, `SignalEnergy`,
`SignalFftAutoScale`) que **não têm membro no enum nem `case` no switch** do
wrapper. Não é limitação de configuração: é ausência estrutural.

O `micro_speech_quantized.tflite` isolado roda, porque seus 4 kernels são
suportados — mas ele recebe features já prontas.

**Serve para:** a cobertura da validação. O audio preprocessor é o único modelo
do conjunto que exercita duas coisas que o desenho precisa suportar — custom ops
e pipeline de dois estágios. Deixá-lo de fora reduz a cobertura em mais do que
um modelo a menos.

Custo de incluir, medido: **nenhuma biblioteca extra**. Os kernels de sinal já
entram na `libtensorflow-microlite.a`
(`tools/make/Makefile:419`: `MICROLITE_CC_KERNEL_SRCS += $(MICROLITE_CC_SIGNAL_KERNEL_SRCS)`),
e puxam kissfft, que o `third_party_downloads` já baixa.

### 5.4.4 Multi-tensor não aparece em modelo real algum

Dos 144 arquivos `.tflite` do repositório do TFLM, **127 são fixtures de
operador único** em `integration_tests/seanet` (um ADD, um PAD isolado). Sobram
17 modelos de verdade, e **todos têm uma entrada e uma saída** — os dois únicos
arquivos com duas entradas também são fixtures (`simple_add_model`).

**Serve para:** o suporte a múltiplos tensores se justifica pelo formato
(`SubGraph.inputs`/`outputs` são vetores no schema) e por modelos de detecção de
objetos, que produzem 4 saídas via `TFLite_Detection_PostProcess` — op que o
TFLM implementa mas para o qual não distribui nenhum modelo. **Não** se
justifica por frequência observada. Registrar isso evita uma afirmação frágil no
texto.

### 5.4.5 Biblioteca cresce, firmware encolhe — medir o lugar certo

Efeito colateral da decomposição: o **objeto do wrapper cresceu** de 56.664 para
147.432 bytes. Antes, os `#if` faziam o compilador descartar 93 dos 98 kernels
já na compilação; agora todos são compilados, e a seleção passou para a
linkedição.

Medindo onde importa — um programa que usa **um** kernel, linkado contra essa
biblioteca de 147 KB:

| | `.text` do binário final |
|---|---|
| sem `--gc-sections` | 42.973 B |
| com `--gc-sections` | **8.024 B** |

**−81%.** E apenas 1 das 98 funções `tflm_add_*` sobrevive no binário.

**Serve para:** é um alerta metodológico para a 5.4. Medir o tamanho do `.a` ou
do `.o` do wrapper **induz a conclusão oposta à verdadeira**. A métrica válida é
o firmware linkado. Vale declarar isso explicitamente no protocolo experimental,
porque um revisor pode questionar por que a biblioteca ficou maior.

### 5.4.6 Primeira execução real — a refatoração funciona

Até 2026-09-28 nada havia sido executado. Validação em duas camadas, para
isolar a origem de uma falha:

**Camada 1 — teste C++ direto contra o wrapper** (sem o robcmp no meio). Varre
x em [0, 2π] em 41 pontos e compara a predição com `sin(x)`:

| Modelo | Erro máximo contra `sin(x)` |
|---|---|
| `hello_world_float` | 0,0394 |
| `hello_world_int8` | 0,1161 (passa pela quantização unificada no B6) |

Também exercitados e aprovados: 10 ciclos create/free (slots são reciclados),
limite de 4 modelos simultâneos com falha controlada no 5º, e handles nulos em
todas as funções sem crash. Os três suspeitos listados no roteiro — `in_use`
não resetado, alinhamento do `interpreter_storage`, conversão de handle —
ficam descartados.

**Camada 2 — programa RL com `use ai.tflm`**, compilado pelo robcmp para o
host (sem `-a`, como o TCC fazia), linkado com o wrapper e a
`libtensorflow-microlite.a`. Saída: `sin(1,57)` → 0,995514, `sin(0)` → 0,026405.

### 5.4.7 `TF_LITE_STATIC_MEMORY` altera o layout de structs

O TFLM compila com `-DTF_LITE_STATIC_MEMORY`, e isso muda o tamanho de
`TfLiteTensor`:

```
sizeof(TfLiteTensor) sem o define = 112 bytes
sizeof(TfLiteTensor) com o define =  64 bytes
```

Um wrapper compilado sem o define lê `data`, `type` e `params` em offsets
errados: lixo silencioso, sem erro de compilação ou de linkedição. O Makefile do
TCC já passava a flag; **o Makefile da Etapa 4 precisa herdá-la**. Flags do
wrapper usadas na validação:
`-std=c++17 -DTF_LITE_STATIC_MEMORY -fno-rtti -fno-exceptions -ffunction-sections -fdata-sections`.

### 5.4.8 `PrintFloat` não imprimia nada — corrigido

O wrapper desligava os logs redefinindo o próprio `MicroPrintf` como no-op, e o
`PrintFloat` — função do usuário, não log de diagnóstico — era implementado com
`MicroPrintf`. Com logs desligados, `PrintFloat` sumia. No TCC era igual (o
`kernel_config.h` vinha com logs desligados); os testes de lá dependiam só do
código de saída, e por isso ninguém notou.

Correção: logs de diagnóstico passaram a usar um macro próprio, `TFLM_LOG`, e
`MicroPrintf` ficou intacto.

### 5.4.9 Literal decimal corrompe o modelo embutido no modo `ai.tflm`

Buraco de tipagem na RL, fora do escopo de TinyML mas com efeito direto sobre
ele. Um literal com valor acima de 127 em decimal é inferido como `int16`:

```
{0x1c, 0x0, 0xff}  ->  [3 x i8]
{28, 0, 255}       ->  [3 x i16]
```

e o vetor `int16` é passado a um parâmetro `uint8[]` **sem conversão e sem
aviso** — a função recebe o ponteiro cru e lê os bytes intercalados com zeros.
Quem embutir os bytes de um `.tflite` em decimal obtém um modelo corrompido em
silêncio. Em hexadecimal, como o TCC fazia, funciona.

**Serve para:** um argumento a favor da sintaxe nativa (5.2) — nela o compilador
lê o `.tflite` pelo caminho, e o usuário nunca escreve esses bytes. Registrar
como limitação do modo biblioteca. Não corrigido: é o verificador de tipos do
compilador base.

### 5.4.10 Alinhamento do modelo e da arena depende de heurística do backend

O TFLM exige arena alinhada a 16 bytes (`MicroArenaBufferAlignment()`), e lê
pesos `float` direto do buffer do modelo. Os globais que a RL emite para o
modelo e a arena **não têm atributo `align`** no IR. Medido nos objetos, ambos
saem alinhados a 16 em x86_64 e em stm32f1 — porque o backend LLVM aplica esse
alinhamento a arrays grandes por conta própria, não por garantia da linguagem.

**Serve para:** risco a verificar na Etapa 4. No Cortex-M4, carga `float`
desalinhada (VLDR) gera falha de barramento; se algum dia o backend deixar de
alinhar, o sintoma aparece só no hardware.

### 5.4.11 Teste diferencial: wrapper idêntico ao TFLM puro nos cinco modelos

Critério: para a mesma entrada, o wrapper e o TFLM usado diretamente produzem
**os mesmos bytes** de saída. Separa "o wrapper está correto" de "o modelo
acerta", verificado à parte. Reproduzir: `make -C test/tflm check`.

**Escopo — importante:** o teste diferencial é feito em C++
(`test/tflm/validate_wrapper.cpp`), chamando as funções `extern "C"` do wrapper
diretamente. Ele prova que o wrapper está correto, mas **não** exercita o código
que o robcmp gera para chamá-lo (passagem de vetores com tamanho injetado,
conversões de tipo da RL, laços). O caminho RL é validado à parte, pelos testes do TCC portados
(5.4.19).

| Modelo | Diferencial | Resultado do modelo |
|---|---|---|
| hello_world_float | — | erro máximo 0,0394 contra `sin(x)`, 41 pontos |
| hello_world_int8 | — | erro máximo 0,1161 |
| spam | idêntico | frase legítima: 97,6% legítima; spam: 91,7% spam |
| person_detect | idêntico | com pessoa: 94,1%; sem pessoa: 72,3% |
| micro_speech (preprocessor) | **40/40 features idênticas ao teste oficial** | valida os 9 custom ops |
| micro_speech (pipeline) | — | "yes" → yes; "no" → no |

No person_detect, a imagem de entrada (já em int8 quantizado) passa pela API
float — desquantizada no teste e requantizada no wrapper — e a ida e volta
reconstrói os 9216 bytes exatos.

### 5.4.12 Arena subdimensionada derruba o TFLM — janelas de crash

O TFLM não trata falta de memória em dois pontos, e cai em vez de devolver erro:

1. **`MicroAllocator::Create`**: coloca três estruturas de controle na arena
   (`SingleArenaBufferAllocator`, `GreedyMemoryPlanner`, `MicroAllocator`) em
   alocações seguidas, sem verificar o resultado — só há `TFLITE_DCHECK`,
   desligado em release. Arena pequena demais: objeto construído sobre `nullptr`.
2. **`AllocatePOD`** (`tensorflow/lite/core/api/flatbuffer_conversions.h:47`),
   usado por **todo parser de operador**:
   ```cpp
   void* allocated_memory = this->Allocate(sizeof(T), alignof(T));
   return new (allocated_memory) T();   // T() zera a struct no endereco 0
   ```
   O parser até verifica `TF_LITE_ENSURE(params != nullptr)`, mas depois do
   `T()` já ter escrito no endereço nulo.

Mapa medido no `hello_world`, cada tamanho num processo isolado:

| Arena | Resultado |
|---|---|
| 0 – 864 B | falha controlada |
| **872 – 976 B** | **crash dentro do TFLM** |
| 984 – 1600 B | falha controlada |
| 1608 B + | funciona |

O wrapper passou a recusar arenas abaixo do mínimo das estruturas de controle
(`kArenaMinima`, calculado a partir dos `sizeof` reais), o que cobre o ponto 1.
O ponto 2 não é interceptável de fora: a janela de 872–976 B continua derrubando
o programa. A correção é uma linha no `AllocatePOD`
(`if (!allocated_memory) return nullptr;`), candidata a PR no TFLM. O submódulo
não foi alterado.

O wrapper do TCC tinha o mesmo problema: sua verificação `if (!allocator)`
nunca disparava, porque o crash acontece antes dela.

**Serve para:** o argumento mais forte a favor de derivar a arena em tempo de
compilação (5.2). Arena mal dimensionada é o erro de configuração mais comum em
TinyML, e o TFLM não o reporta de forma confiável — em certas faixas, o programa
simplesmente cai. Deixa de ser conveniência e passa a ser **segurança**.

### 5.4.13 Lacuna na API: não há caminho para dados já quantizados

A API do wrapper só aceita e devolve `float`, e quantiza na escrita. No pipeline
do micro speech, as features saem do preprocessor como códigos `int8` **sem**
parâmetros de quantização, e o classificador as espera como `int8` **com** scale
0,101716. O exemplo oficial copia os bytes crus. Pela API float, as features são
requantizadas:

| Áudio | Bytes da entrada alterados | Saturados em ±127/−128 |
|---|---|---|
| "yes" | 767 de 1960 (39%) | 1893 |
| "no" | 152 de 1960 | 1946 |

O rótulo ainda saiu certo — um espectrograma "binarizado" basta para distinguir
yes de no —, o que **escondia** a corrupção até a medição direta. Com outro
áudio ou outro modelo, erraria.

Casos afetados: pipeline entre modelos (saída de um é entrada do outro) e dados
de sensor já na representação do modelo (imagem int8).

**Resolvido** com acesso cru por tipo, sem conversão:

```c
int8_t SetTensorRawInt8 (uintptr_t, const int8_t*,  int32_t len);
int8_t GetTensorRawInt8 (uintptr_t,       int8_t*,  int32_t len);
int8_t SetTensorRawInt16(uintptr_t, const int16_t*, int32_t len);
int8_t GetTensorRawInt16(uintptr_t,       int16_t*, int32_t len);
```

Exigem tipo do tensor igual ao do vetor e tamanho igual ao número de elementos
do tensor; senão devolvem −1 **sem copiar nada** (testado). Com elas, o pipeline
do micro speech roda como o exemplo oficial, e a imagem do person_detect entra
direto, sem a ida e volta pelo float.

Duas decisões de desenho, ambas por evidência:

- **Uma cópia crua tensor-a-tensor não resolveria o micro speech**: o
  preprocessor produz 40 features por chamada, que vão para posições diferentes
  dos 1960 da entrada do classificador. Os tamanhos diferem. O que resolve é
  leitura/escrita crua por vetor, com o programa montando o buffer.
- **Separadas por tipo, não genéricas por bytes**: a RL injeta a contagem de
  *elementos* do vetor, não de bytes. Uma versão por bytes recebendo `int16[]`
  copiaria só metade dos dados — e o buraco de tipagem (5.4.9) deixaria passar
  sem aviso.

Na RL, um vetor `int16` se declara por conversão explícita: `{int16(0):480}`.
Literais inteiros pequenos inferem `int8`.

**Serve para:** a 5.2 — `classifier.input[0] = pre.output[0]` na sintaxe nativa
precisa copiar crus quando os tipos batem. Com os tipos conhecidos em tempo de
compilação, o compilador sabe escolher entre cópia crua e conversão.

### 5.4.14 `use` não exporta variáveis globais nesta branch

Um arquivo incluído por `use` tem suas declarações de função visíveis — é o que
faz `use ai.tflm` funcionar —, mas **não** suas variáveis globais:
`Symbol dados_modelo not found`. Mesmo sintoma do `use math.float` quebrado
visto no início (constante `NAN` não encontrada). Consequência prática: os
bytes de um modelo não podem ficar num módulo separado; o gerador
`test/tflm/tflite_to_rob.py` insere o vetor no próprio programa. Não
investigado: é comportamento da base `cola2025`.

### 5.4.15 Kernels especializados reduziriam o binário

O teste oficial do person_detect registra variantes int8 dos kernels
(`Register_CONV_2D_INT8()`, `Register_SOFTMAX_INT8()`,
`Register_AVERAGE_POOL_2D_INT8()`). O wrapper usa o registro genérico
(`AddConv2D()`), que inclui também o caminho float. Mesmo resultado numérico,
binário maior. Com o tipo dos tensores conhecido em tempo de compilação (5.2),
o compilador pode escolher a variante especializada.

**Serve para:** medição de tamanho de binário na 5.4 e argumento adicional para
a análise estática.

### 5.4.16 Defeito latente corrigido: acesso ao resolver pelo handle

`resolver_from_handle` fazia `reinterpret_cast<MutableResolver*>(h)`, mas o
handle aponta para a `TFLM_Instance`. Funcionava só porque `resolver` era o
primeiro campo da struct. Reordená-la faria as 119 funções `tflm_add_*`
escreverem no lugar errado, sem erro. Corrigido para `&instance->resolver`.

### 5.4.17 Regressão introduzida no B6 e corrigida: `SetTensorFromIntArray`

No wrapper do TCC, `SetTensorFromIntArray` era uma **cópia crua**: cada valor ia
para o tipo do tensor por `static_cast`, sem quantizar. A unificação da
conversão de tipos no B6 passou a rotear essa função por `tensor_write`, que
quantiza. O teste de micro speech do TCC passa por ela códigos int8 já prontos,
que seriam corrompidos.

Não foi pego pela camada 1 porque essa função não era exercitada contra o
comportamento original. Descoberto ao portar os testes do TCC (2.7), comparando
tipo a tipo a semântica das funções de conversão do TCC com as do B6.

Corrigido restaurando a cópia crua, e acrescentado a `validate_wrapper.cpp` o
caso que teria pego a regressão: o valor 5 escrito num tensor int8 com scale
0,0244801 precisa chegar como 5.

As demais diferenças de semântica entre o TCC e o B6 só aparecem em casos que
nenhum modelo de validação exercita, e foram mantidas por serem mais corretas:
tensor `int32`/`int64` com parâmetros de quantização (o TCC os ignorava; o B6
aplica) e `uint8` fora da faixa (o TCC deixava dar a volta; o B6 satura).

### 5.4.18 O teste de person detection do TCC não entregava a imagem ao modelo

**Achado que afeta o baseline.** No `lib-person-detection.rob` do TCC, a imagem
é declarada em hexadecimal — a RL infere `int8[]` — e passada a
`SetTensorFromIntArray`, cujo parâmetro é `int16[]`. Pelo buraco de tipagem da
5.4.9, o ponteiro segue sem conversão, com o tamanho 9216. A função lê então
9216 valores de 16 bits: **18432 bytes de um vetor de 9216**.

Mecanismo medido, reproduzindo a chamada e lendo o tensor de volta com
`GetTensorRawInt8`:

| Imagem | Bytes do tensor ≠ imagem | `tensor[i] == imagem[2i]`, i < 4608 |
|---|---|---|
| com pessoa | 9158 / 9216 | 4608 / 4608 |
| sem pessoa | 9177 / 9216 | 4608 / 4608 |

A primeira metade do tensor recebe um pixel sim, um não, da imagem inteira; a
segunda metade é **memória além do fim do vetor**.

Reproduzindo o programa do TCC com fidelidade — mesma ordem de declaração dos
globais, mesma chamada —, 16 execuções do mesmo binário:

| | p(pessoa) no teste "com pessoa" | p(pessoa) no teste "sem pessoa" |
|---|---|---|
| entrada correta (`SetTensorRawInt8`) | 0,9414 | 0,2773 |
| estilo TCC, 16 execuções | 0,6133 em todas | **de 0,246 a 0,508** |
| estilo TCC, ASLR desligado | 0,6133 | 0,2500 em todas |

- A imagem com pessoa é determinística porque a leitura além do fim cai no
  vetor declarado em seguida — a imagem sem pessoa.
- A imagem sem pessoa **varia a cada execução**: a leitura além do fim entra na
  arena, que guarda ponteiros, e os endereços mudam com o ASLR. Numa das 16
  execuções, p(pessoa) = 0,508 > 0,492 — a imagem **sem pessoa foi classificada
  como pessoa** e o teste falharia.

Conclusão: **o teste de person detection do TCC era intermitente**, e quando
passava, passava por coincidência — o resultado dependia da disposição da
memória, não da imagem.

**Serve para:**
- a 5.4, que precisa declarar que a validação de person detection do TCC não é
  válida como baseline de corretude. Tamanho de binário e, provavelmente, tempo
  de inferência não são afetados — uma CNN faz o mesmo número de operações
  qualquer que seja o conteúdo da entrada —, mas isso deve ser dito com cuidado;
- o argumento a favor da sintaxe nativa: o erro nasce de o usuário escolher à
  mão a função de escrita e o tipo do vetor, e de a RL não verificar a
  compatibilidade. Na sintaxe nativa, o compilador conhece o tipo do tensor e a
  visão `.raw` rejeita a incompatibilidade em tempo de compilação (5.2.6);
- o buraco de tipagem da 5.4.9 deixa de ser hipotético: tem uma vítima medida.

### 5.4.19 Testes do TCC portados, em RL: todos passam

Os cinco testes do TCC foram portados para a API atual (`test/tflm/rob/tcc_*.rob`)
mantendo lógica, entradas, limiares e códigos de retorno. Compilados pelo robcmp
para o host, linkados com o wrapper e o TFLM:

| Teste | Resultado |
|---|---|
| `tcc_hello_world_float` | OK |
| `tcc_hello_world_int8` | OK |
| `tcc_spam` | OK — p(spam) 0,023690 e 0,916583, os mesmos da camada C++ |
| `tcc_person_detect` | OK — com a imagem entregue corretamente (`SetTensorRawInt8`) |
| `tcc_micro_speech` | OK |

Os testes conseguem falhar: trocando as imagens entre si, `tcc_person_detect`
falha com código 1; trocando as features de "yes" e "no", `tcc_micro_speech`
falha com código 3.

**Equivalência de valores entre RL e C++.** Os testes do TCC só devolvem
passou/falhou contra um limiar, então passar não prova que os valores batem.
Verificado à parte: os mesmos 24 casos (hello_world float e int8 com x = 0,0 e
1,57; as duas frases do spam; as duas imagens do person_detect; os três áudios
do micro speech, cada classe) rodados por um programa C++ e por variantes dos
programas RL que imprimem cada saída. **24 de 24 idênticos**, até a sexta casa
decimal. Como os dois lados chamam o mesmo wrapper, isso confirma que o robcmp
passa vetores, tamanhos injetados e floats sem alteração.

Nota prática: no x86, o `PrintFloat` escreve na saída de erro (`stderr`), porque
o log do TFLM (`DebugLog`) vai para lá. Quem capturar a saída de um programa RL
precisa redirecionar `2>&1`.

Os dados de teste ficam em `test/tflm/data/`, com a origem documentada
(`data/README.md`); as imagens do TCC conferem byte a byte com as do TFLM.
Reproduzir: `make -C test/tflm check`.

---

## Registro de modificações

### 2026-09-14 — `efad4af` Correção de build herdada do upstream

O commit `d3dfe43` renomeou a classe `Loop` para `InfinityLoop` em `Loop.h` e
`Loop.cpp` mas não atualizou a ação da gramática em `Language.y:392`, que
continuava instanciando o nome antigo. A branch base **não compilava**.

Candidato a PR para o upstream (`thborges/robcmp`) — afeta qualquer um na
`cola2025`.

### 2026-09-14 — `cb94af8` TFLM como submódulo

Fixado em `d0318206`. O TFLM não publica tags nem releases, então `main` é o
canal de versionamento. Caminho `third-party/tflite-micro`, igual ao da branch
do TCC, para os includes do wrapper resolverem sem edição.

### 2026-09-21 — `e1f108a` Wrapper, modelos e inspetor estático

Wrapper importado do TCC como baseline. `kernel_config.h` auditado (ver 5.1.1).
Sete modelos de validação em `test/tflm/models/`. Criado
`test/tflm/tflite_info.cpp`, inspetor estático de `.tflite` e esqueleto do
módulo da 5.2.

### 2026-09-21 — Refatoração do wrapper, etapa B4 + C9 + C8

- **B4**: removidas `DiagnoseModel` e `VerifyModelData` do `.cpp` e as
  declarações de `AnalyzeModelKernels`, `DiagnoseModel` e `VerifyModelData` do
  `.h`. Verificado antes de remover que nenhum `.rob` as chama — aparecem apenas
  como declaração na fachada `lib/ai/tflm.rob` do TCC. Resultado em 5.4.1.
- **C9**: removido o `case RANDOM_STANDARD_NORMAL`. Entrada do enum mantida (os
  valores espelham `BuiltinOperator`; alterá-los quebraria a numeração) com
  anotação do motivo.
- **C8**: 6 warnings → 0. Três estavam dentro do `DiagnoseModel`. Dos demais,
  `builtin_op` existia só para log e foi inlinado; os outros dois vinham de um
  bloco "TENSOR DEBUG" com dois laços cujo corpo inteiro era `MicroPrintf` —
  percorriam todos os tensores sem fazer nada com log desligado.

Pendência deixada de propósito: `RunBenchmarkOptimized` segue declarado sem
definição (item C7, decisão pendente).

### 2026-09-23 — Refatoração do wrapper, etapas A1, A2, A3 e B5

**A1** — `RegisterOp` (700 linhas, 98 `case`) decomposto em 98 funções
independentes `tflm_add_<kernel>(uintptr_t) -> int8_t`, geradas
programaticamente a partir do `switch` existente (extração dos pares
`case → resolver->AddXxx()`, verificada contra duplicatas e argumentos).

**A2** — apagados o `kernel_config.h` (269 linhas), o enum `KernelType` (213
linhas no `.cpp` e outro tanto no `.h`), o despachante `RegisterOp` e
`MapBuiltinOperatorToKernelType`. `ENABLE_DEBUG_LOGS` virou `TFLM_DEBUG_LOGS`,
controlável por `-D` na build em vez de edição de arquivo.

**A3** — `InitializeInterpreter` substituído pelo ciclo em três fases:
`tflm_create_resolver()` → `tflm_add_*()` → `tflm_create_model()`, mais
`tflm_free_model()`. Assinaturas corrigidas conforme 5.1.5: sem parâmetro
fantasma, tamanhos como `int32_t`.

**B5** — heap eliminado. Instâncias em `g_instances[TFLM_MAX_MODELS]`
(armazenamento estático), `MicroInterpreter` por *placement new*.
`AllocAndGetTensorArray` removida (`malloc`, sem uso em nenhum `.rob`).

Totais: `.cpp` 1787 → 1356 linhas; `.h` 299 → 202; `kernel_config.h` 269 → 0.
De 2355 para 1558 linhas, **−34%**. Resultado de binário em 5.4.5.

Pendências mantidas: `RunBenchmarkOptimized` segue declarado sem definição
(C7); `TFLM_MAX_OPS` fixado em 128 (C10 — ver restrição abaixo).

**Restrição descoberta em C10:** `MicroMutableOpResolver<tOpCount>` é template,
logo `tOpCount` é fixado quando o wrapper vira `.a`. Como a 5.3 exige distribuir
o wrapper pré-compilado, esse valor **não pode** ser derivado por programa pela
análise estática. Saídas possíveis: valor fixo generoso (adotado: 128, cobre os
121 kernels do TFLM) ou instanciar algumas variantes de tamanho.

### 2026-09-23 (continuação) — B6, C7, C11, cobertura de kernels e assinaturas

**21 kernels acrescentados** (não 19 como estimado): os 19 custom ops mais
`AddReduceAll` e `AddDynamicUpdateSlice`, os dois builtins cujas anotações no
`kernel_config.h` estavam desatualizadas. Cobertura passa a 119 — tudo o que o
TFLM expõe, exceto `AddBuiltin`/`AddCustom`, que são mecanismos e não kernels.
Os nove `Signal*` desbloqueiam o `audio_preprocessor` e, com ele, o pipeline
completo do Micro Speech.

**C7** — `RunBenchmarkOptimized` removida. Nunca foi implementada.

**C11** — `InvokeInterpreter` passa a devolver `int8_t` (0 = sucesso). O status
do `interpreter->Invoke()` já era capturado e verificado, e ia para um
`MicroPrintf` que com log desligado é no-op: a falha era engolida por completo.

**B6** — camada de conversão unificada em `tensor_write`/`tensor_read`:

| | antes | depois |
|---|---|---|
| `SetTensorValue` | 222 | 4 |
| `GetTensorAsFloat` | 126 | 3 |
| `SetTensorFromIntArray` | 77 | 5 |
| `GetTensorSize` | 45 | 8 |
| `SetTensorArray` | 19 | 5 |
| `GetTensorArray` | 16 | 5 |

Seis `switch(tensor->type)` viraram dois, ambos no ponto único de despacho.

Dois defeitos que a duplicação escondia, corrigidos no processo:

1. **Leitura cobria menos tipos que a escrita** — 4 contra 6; faltavam `int16` e
   `int64`. Era possível escrever num tensor `int16` e não conseguir lê-lo, e a
   entrada do `audio_preprocessor` é justamente `int16`.
2. **`GetTensorSize` podia errar** — calculava `bytes / tamanho_do_elemento` com
   switch próprio. Trocado por `tflite::ElementCount(*t->dims)`, helper do
   próprio TFLM, que deriva o total das dimensões.

O *clamping* também só existia em alguns ramos; agora é uniforme.

**Parâmetro fantasma corrigido** (achado 5.1.5). Sete funções declaravam um
`int` final sem ter argumento vetor — a RL nunca o passava. Verificado no IR
emitido:

```
antes:  declare i64 @GetInputTensor(i64, i64)   vs   C: (uintptr_t, size_t, int)
depois: declare i64 @GetInputTensor(i64, i64)   vs   C: (uintptr_t, size_t)
```

Nas três que têm vetor (`SetTensorArray`, `SetTensorFromIntArray`,
`GetTensorArray`) o parâmetro é real e foi mantido, agora com nome
(`values_len`) e comentário explicando que a RL o injeta.

**`lib/ai/tflm.rob` regenerado** a partir do header, com os 119 kernels
agrupados por natureza (builtin / sinal / outros custom) e as assinaturas novas.

Totais da refatoração completa: `.cpp` 1787 → 1175, `.h` 299 → 212,
`kernel_config.h` 269 → 0. **2355 → 1387 linhas, −41%**, com 20 kernels a mais
expostos.

---

## Roteiro — o que foi feito e o que vem

### ✅ Etapa 1 — Refatoração do wrapper (concluída em 2026-09-23)

| Item | Descrição | Estado |
|---|---|---|
| A1 | Decompor `RegisterOp` em funções por kernel | feito |
| A2 | Eliminar enum `KernelType` e `kernel_config.h` | feito |
| A3 | Sequência de inicialização em três fases | feito |
| B4 | Remover análise em runtime | feito |
| B5 | Eliminar heap (13 `new`/`delete` + 1 `malloc`) | feito |
| B6 | Unificar 6 `switch(tensor->type)` duplicados | feito |
| C7 | `RunBenchmarkOptimized` | removida |
| C8 | Warnings de variável não usada | feito |
| C9 | `AddRandomStandardNormal` inexistente | feito |
| C10 | `TFLM_MAX_OPS` derivado da análise estática | **inviável** (template), fixado em 128 |
| C11 | `invoke()` devolver status | feito |
| — | Expor os 21 kernels ausentes | feito, cobertura 119 |
| — | Parâmetro fantasma em 7 assinaturas | feito |
| — | Adaptar `lib/ai/tflm.rob` | feito |

Resultado: 2355 → 1387 linhas (−41%), 99 → 119 kernels, heap zerado.

---

### ✅ Etapa 2 — Validação (concluída em 2026-09-28)

**Situação crítica a registrar: nada foi executado até aqui.** Toda a
verificação da Etapa 1 foi em nível de compilação e símbolos. A
`libtensorflow-microlite.a` nunca foi compilada e nenhuma inferência rodou.

Ordem sugerida, do menor risco para o maior:

| # | Tarefa | Por quê nesta ordem |
|---|---|---|
| 2.1 | Compilar `libtensorflow-microlite.a` (x86_64) | **feito** — 2,3 MB, `gen/linux_x86_64_default_gcc/lib/` |
| 2.2 | Linkar e rodar `hello_world_float` | **feito** — camadas 1 e 2 passam (ver 5.4.6) |
| 2.3 | Rodar `hello_world_int8` | **feito** — erro máximo 0,1161 |
| 2.4 | Rodar `spam` | **feito** — idêntico ao TFLM puro (5.4.11) |
| 2.5 | Rodar `person_detect` | **feito** — idêntico ao TFLM puro |
| 2.6 | Rodar `micro_speech` + `audio_preprocessor` | **feito** — preprocessor 40/40; expôs a lacuna 5.4.13 |
| 2.7 | Adaptar os 5 `.rob` de teste do TCC à API nova | **feito** — 5/5 passam em RL (5.4.19); revelou 5.4.17 e 5.4.18 |
| 2.8 | Acrescentar à API caminho para dados já quantizados | **feito** — `SetTensorRaw*`/`GetTensorRaw*` (5.4.13) |
| 2.9 | Correção do `AllocatePOD` | **adiado** — ver abaixo |

Sobre o 2.9: o `flatbuffer_conversions.h` é copiado pelo TFLM do repositório
principal do TensorFlow (`ci/tflite_files.txt`), então a correção teria de ir
para `tensorflow/tensorflow`, não para o TFLM — um PR no TFLM seria sobrescrito
na próxima sincronização. O bug segue presente no `master` do TensorFlow. É
comportamento indefinido pelo padrão C++ (CWG 1748: *placement new* sobre nulo),
e a correção é uma linha. Não bloqueia nada: só ocorre com arena mal
dimensionada numa faixa estreita. Adiado por custo (CLA, revisão lenta num
repositório enorme); o valor para a dissertação — o argumento de segurança —
já está medido.

**Prioridade do 2.7:** é o único item que valida o caminho RL dos modelos além
do `hello_world`. Não é detalhe.

Tudo roda com `make -C test/tflm check` (camadas 1 e 2).

Comando para 2.1:

```bash
make -C third-party/tflite-micro -f tensorflow/lite/micro/tools/make/Makefile microlite
```

O que investigar primeiro se algo falhar: a Etapa 1 trocou construção dinâmica
por *placement new* (`g_instances`), mudou toda a API de ciclo de vida e
reescreveu a conversão de tipos. Suspeitos na ordem: `in_use` não resetado,
alinhamento do `interpreter_storage`, conversão de handle.

---

### ⬜ Etapa 3 — Compilador, seção 5.2 (PRÓXIMA)

| # | Tarefa |
|---|---|
| 3.1 | Migrar `test/tflm/tflite_info.cpp` para módulo em `src/` (fase de elaboração prévia) |
| 3.2 | Acrescentar as produções P1 e P2 à gramática (ver 5.2.1: zero conflitos medidos) |
| 3.3 | Tipo `ai_model` na semântica, com a regra de precedência que preserva função do usuário |
| 3.4 | Tipagem de `output[i]` derivada do `.tflite` |
| 3.5 | Emitir as chamadas `tflm_add_*` conforme os kernels detectados |
| 3.6 | Erros de compilação: shape incompatível, kernel ausente, caminho não-literal, índice não-constante |
| 3.7 | Fechar as decisões em aberto de `test/tflm/native-syntax.rob` (seção 9) |

A especificação executável da sintaxe está em `test/tflm/native-syntax.rob`.

---

### ⬜ Etapa 4 — Build unificado, seção 5.3

| # | Tarefa |
|---|---|
| 4.1 | `stm32f407.ld` — mapa de memória do STM32F407VET6 |
| 4.2 | `stm32f407.rob` — periféricos implementando `intfs.mcu`, incluindo o registrador DWT |
| 4.3 | Estender `platformio/platform/builder/main.py` |
| 4.4 | Distribuir `.a` pré-compiladas: x86_64 e ARM Cortex-M4 |

---

### ⬜ Etapa 5 — Experimentação, seção 5.4

| # | Tarefa |
|---|---|
| 5.1 | Rodar os modelos nos dois modos (`ai.tflm` e sintaxe nativa) |
| 5.2 | Rodar nas duas plataformas (x86_64 e STM32F407VET6) |
| 5.3 | Coletar tamanho de binário e tempo (Hyperfine no x86, DWT no STM32) |
| 5.4 | Comparar contra implementação equivalente em C++ puro com TFLM |

Atenção metodológica registrada em 5.4.5: **medir o firmware linkado, não o
`.a`** — a biblioteca cresceu com a refatoração enquanto o binário encolheu 81%.

---

## Decisões ainda em aberto

- Nome do construtor na sintaxe nativa: `ai_model` foi aplicado em
  `native-syntax.rob`, aguardando confirmação do orientador.
- `arena_size`: derivar exige reproduzir estaticamente o memory planner do TFLM.
  Enquanto não existir, o argumento explícito é a saída.
- `input.count` / `output.count` existem na sintaxe?
- Modelos com mais de um subgrafo: ignorar os demais ou recusar com erro?

---

## Estado do repositório

Em 2026-09-23, a Etapa 1 estava **concluída e não commitada**:

```
 D wrappers/tflm/kernel_config.h
 M wrappers/tflm/tflm_wrapper.cpp
 M wrappers/tflm/tflm_wrapper.h
?? lib/ai/tflm.rob
?? RESULTADOS.md
?? test/tflm/native-syntax.rob
```

Último commit: `e1f108a` — wrapper do TCC, modelos e inspetor estático.
