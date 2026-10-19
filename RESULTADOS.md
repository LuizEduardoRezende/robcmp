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

### 5.2.1 O desenho indexado exigia três produções novas de gramática

**Substituído (2026-10-05, ver 5.2.12).** Este registro mede o desenho em que
os tensores eram indexados (`m.input[0]`). O desenho adotado, com membros
nomeados (`m.input0`), não exige nenhuma produção. A medição continua valendo
como termo de comparação entre os dois.


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
`src/semantic/PropagateTypes.cpp:344`, conforme o nome seja tipo ou função.

É nesse mesmo ponto que a 3.3 reconhece `ai_model(...)`
(`PropagateTypes.cpp:324`). A regra de precedência prevista aqui — função do
usuário vence o embutido — foi trocada pela reserva do nome (5.2.13).

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

### 5.2.7 O robcmp não marca `signext` em inteiros pequenos com sinal

**Correção (2026-10-06).** A versão anterior deste registro dizia que o robcmp
declara `zeroext i8` nos retornos `int8`. Está errado: o `check_abi.py` agrupa
os atributos da assinatura inteira, e o `zeroext` visto era o do parâmetro
`uint64`. Lido no IR, função por função:

```
RL (robcmp):  declare i8 @ret_i8()            declare i32 @amplia(i8)
C  (clang):   define signext i8 @ret_i8()     define i32 @amplia(i8 signext %0)
```

O robcmp põe `zeroext` nos tipos sem sinal e **nada** nos com sinal
(`src/FunctionDecl.cpp:150` e `:158`, `src/FunctionImpl.cpp:52` e `:86`). Falta o
`signext` em `int8` e `int16`, em parâmetros e retornos.

**Não é risco latente: produz valor errado.** A RL passa −1 (`int8`) a uma função
C que o devolve como `int32`:

| Lado C compilado por | Código da função | Resultado |
|---|---|---|
| clang (x86_64) | `mov %edi,%eax` — confia na extensão feita por quem chama | **255** |
| gcc (x86_64) | `movsbl %dil,%eax` — estende de novo | −1 |
| clang (Cortex-M) | `bx lr` — a convenção do ARM obriga quem chama a estender | não executado; mesmo padrão |

Reproduzir: `make -C test/achados signext`.

No sentido inverso (retorno), o caminho RL ← C é seguro: sem o atributo, quem
chama não supõe extensão e estende. Comparações de `tflm_invoke(...)` com `0` e
`-1` acertam, como verificado antes.

A API do wrapper não tem parâmetro `int8`/`int16` escalar, então o TinyML não é
afetado hoje. A linguagem é: qualquer função C com esses parâmetros, compilada
pelo clang, recebe valores errados. **Para a Etapa 3 (3.8):** marcar `signext`
nos tipos com sinal.

### 5.2.8 Arena: quanto cada modelo precisa, e quanto o TCC reservava

Medido em x86_64 com o TFLM puro. "Usada" é o `arena_used_bytes()` depois do
`AllocateTensors`; "mínima viável" é a menor arena com a qual o
`AllocateTensors` passa, obtida por busca binária com cada tentativa num
processo isolado (por causa das janelas de crash da 5.4.12). Medida duas vezes:
com o buffer da arena começando num endereço múltiplo de 16, e 8 bytes depois.
Reproduzir: `make -C test/achados arena` (`test/achados/arena/arena.cpp`).

| Modelo | Usada | Mínima (alinhada) | Mínima (+8 bytes) | Arena do TCC | Sobra no TCC |
|---|---|---|---|---|---|
| hello_world_float | 1216 | 1608 | 1616 | 3000 | 87% |
| hello_world_int8 | 1184 | 1608 | 1616 | 3000 | 87% |
| spam | 2000 | 2200 | 2208 | 4096 | 86% |
| person_detect | 85024 | 85016 | 85024 | 90000 | 6% |
| micro_speech_quantized | 7312 | 7304 | 7312 | 12288 | 68% |
| audio_preprocessor_float | 9136 | 9128 | 9136 | — | — |
| audio_preprocessor_int8 | 9952 | 9944 | 9952 | — | — |

A sobra é calculada sobre a mínima alinhada.

**Correção (2026-10-06).** A primeira versão desta tabela tinha só a coluna
"+8 bytes", sem dizer o alinhamento, e juntava os dois hello_world numa linha
com "usada 1216" — o int8 usa 1184. A ferramenta daquela medição não foi
guardada; a de `test/achados` reproduz exatamente os números antigos quando o
buffer começa 8 bytes fora do alinhamento de 16, o que explica a diferença.

Três fatos:

1. **`arena_used_bytes()` não é o mínimo.** Nos modelos pequenos, o planejamento
   de memória precisa de espaço temporário que é devolvido ao final: o
   hello_world usa 1216 bytes mas só aloca com 1608. Nos modelos maiores, o
   temporário cabe dentro do que é usado, e a mínima fica até 8 bytes abaixo da
   usada (que conta o arredondamento de alinhamento). Um cálculo estático da
   arena que reproduza só o `arena_used_bytes` subdimensionaria os modelos
   pequenos — e cairia na faixa de falha, ou de crash.
2. **O alinhamento do buffer muda a arena necessária.** O TFLM descarta os bytes
   até o próximo múltiplo de 16: com o buffer deslocado em 8, todo modelo
   precisa de 8 bytes a mais. Liga-se à 5.4.10 — o alinhamento que o robcmp dá à
   arena depende de heurística do backend, então o cálculo da arena precisa
   somar a margem de alinhamento ou o compilador precisa garantir o alinhamento.
3. **As arenas do TCC eram superdimensionadas em até 87%.** Num STM32F407, com
   192 KB de RAM, reservar arena por tentativa desperdiça memória que o
   firmware poderia usar para outra coisa.

**Serve para:** a motivação do cálculo automático da arena (decisão da seção 6
de `native-syntax.rob`: obrigatória na primeira versão, calculada depois), e
para definir o critério de corretude desse cálculo — ele precisa prever a
mínima viável, não a usada.

Ressalva: medido em x86_64. No Cortex-M, com ponteiros de 32 bits, as
estruturas internas do TFLM são menores, e os números devem cair um pouco.
Não medido.

### 5.2.9 O verificador do FlatBuffers separa o compilador de um crash

O acesso do FlatBuffers é *zero-copy*: os acessores seguem offsets lidos do
próprio arquivo, sem conferir limites. O `tflite_info` do início da pesquisa só
conferia o identificador `TFL3`. No compilador, isso significaria que um
`.tflite` danificado, citado num `ai_model("...")`, derruba o robcmp.

Medido com `test/tflm/check_tflite.py`, sobre o `hello_world_int8.tflite`, com
o módulo compilado com AddressSanitizer, com e sem `tflite::VerifyModelBuffer`:

| Entrada | Sem verificador | Com verificador |
|---|---|---|
| 128 truncamentos (todo byte de 8 a 63, e a cada 37 até o fim) | **110 falhas de memória**, 18 aceitos como modelo válido | 128 rejeitados |
| 300 arquivos com 1–8 bytes corrompidos (semente fixa) | **64 falhas de memória**, 21 rejeitados | 238 rejeitados, 0 falhas |

As falhas sem verificador são `heap-buffer-overflow` e acesso desalinhado. Os
18 truncados "aceitos" (cortes em 44–52, 1100–1322, 2543 e 2654 bytes, de 2704)
são o caso mais traiçoeiro: o corte não atingiu nenhum campo que a análise lê,
então o compilador aceitaria e embutiria um modelo incompleto, e o erro só
apareceria no dispositivo, ao alocar os tensores. Com o verificador, os 62
corrompidos restantes que passam são modelos estruturalmente válidos (um byte
de peso trocado, por exemplo) — corretos de aceitar, e sem acesso inválido.

**Serve para:** a 5.2, como requisito de robustez de uma fase de elaboração que
lê entrada externa dentro do compilador. Um compilador não pode falhar com
*segfault* por causa de um arquivo de dados do usuário; o erro precisa ser
diagnóstico (`corrupted or truncated TFLite file`).

### 5.2.10 Kernels vêm dos operadores, não da lista de códigos

O `Model.operator_codes` é a tabela de códigos do modelo; os operadores a
referenciam por `opcode_index`. O TFLM resolve kernels **por operador**
(`micro_interpreter.cc:119`), então um código que nenhum operador usa não precisa ser
registrado. O módulo coleta os kernels percorrendo os operadores de todos os
subgrafos — o que também cobre os corpos de `IF`/`WHILE` (5b da especificação).

Nos 7 modelos de `models/`, o resultado é idêntico ao da leitura direta de
`operator_codes` (comparado com o `tflite_info` anterior). A diferença é de
robustez: com a leitura direta, um código órfão deixado por um conversor viraria
kernel linkado sem uso — ou, se sem suporte, um erro de compilação falso.

**Serve para:** a 5.2, na descrição do algoritmo de extração de kernels.

### 5.2.11 O modo de declarações do `use` perde o arquivo inteiro com um literal de array

No modo padrão, o `use` lê a biblioteca por uma gramática simplificada
(`src/LanguageUse.y`), que guarda só as declarações e pula o corpo das funções
por recuperação de erro. Com `-bdep`, lê pela gramática completa e compila a
biblioteca junto (`src/Scanner.cpp:185`).

| Biblioteca | Sem `-bdep` | Com `-bdep` |
|---|---|---|
| três funções, uma com `a = {1.0, 2.0};` no corpo | `Symbol dobro not found` — perde **as três**, inclusive a anterior ao literal | compila |
| a mesma, com `a = 1.0;` no lugar do literal | compila | compila |
| uma variável global e uma função (5.4.14) | `Symbol contador not found` | compila |

Confirmado no robcmp do commit `f6263b0`, anterior à gramática da 3.2: o
problema é da base. O `test/general` compila tudo com `-bdep`, por isso nunca
apareceu nos testes. Reproduzir: `make -C test/achados use`. O mecanismo exato
dentro da gramática simplificada não foi investigado.

**Serve para:** uma biblioteca em RL que use a sintaxe nativa terá
`m.input0 = {…}` no corpo das funções, e compilada separadamente cai neste
caso. Levado à orientação de 2026-10-05 como pergunta: corrigir na branch,
reportar ao upstream ou só documentar.

**Decidido (2026-10-16): só documentar.** Fora do escopo da pesquisa; programas
com a sintaxe nativa compilam com `-bdep`.

### 5.2.12 Com tensores como membros nomeados, a sintaxe nativa não exige nenhuma produção

Proposta do orientador (2026-10-05): cada tensor é um membro do modelo, com
nome e número, em vez de `input[0]`, `input[1]`. Adotado com numeração a partir
de 0: `input0`, `input1`, ..., `output0`, .... O elemento k passa a ser
`m.output0[k]` — índice de vetor comum.

| Forma | Desenho indexado | Desenho com membros |
|---|---|---|
| vetor inteiro | `m.input[0] = {1.0}` — P1, nova | `m.input0 = {1.0}` — já aceita |
| elemento | `m.output[0][5]` — duplo índice (`LoadMatrix`) | `m.output0[5]` — `LoadArray` |
| propriedade | `m.input[0].scale` — P2, nova | `m.input0.scale` — identificador pontilhado |
| visão crua | `m.input[0].raw = v` — P3, nova | `m.input0.raw = v` — já aceita |
| elemento cru | não previsto | `m.output0.raw[1]` — já aceita |
| contagem | `m.input.count` | `m.input_count` |

Verificado: `test/tflm/native-syntax.rob`, reescrito no desenho novo (640
linhas), é lido pela gramática original sem nenhum erro de sintaxe — com um
erro proposital na última linha, para provar que o parser chegou ao fim. As
produções P1–P3 (`4dd5c68`) foram revertidas; a gramática voltou a ser idêntica
à do commit `07af0f3`, e o placar de `test/general` não muda (63 passam).

Outros efeitos:

- **Some uma regra.** O índice do tensor precisava ser constante porque cada
  tensor tem um tipo; com o número no nome, é sempre constante.
- **Numeração a partir de 0.** O orientador propôs começar em 1 (`input1`);
  trocado para 0 por três razões, registradas na seção 0 da especificação: na
  mesma expressão `m.output0[0]` tensor e elemento começam do mesmo número; o
  modo biblioteca já chama o primeiro tensor de `tflm_input(m, 0)`; e o TFLite,
  o TFLM e suas ferramentas numeram a partir de 0.
- **Contadores no modelo.** `input_count`/`output_count` contam tensores; `.size`
  conta elementos de um tensor. Por isso não existe `input0.count`.
- **Sem atalho para tensor único.** Não há `m.input` significando "o único
  tensor": mesmo num modelo de uma entrada, ela é `input0` (decisão da seção 5).

**Serve para:** a 5.2, como o resultado mais forte possível para a
classificação de extensão minimamente intrusiva (Mernik/Heering/Sloane):
**nenhuma mudança sintática**, todo o trabalho na semântica. A comparação com o
desenho indexado (três produções, 5.2.1) mostra que a escolha de desenho, e não
só a implementação, determina o quanto a extensão é intrusiva.

### 5.2.13 `ai_model` é nome reservado — na semântica, não no léxico

**O problema.** A especificação previa (seção 7a) que uma função do usuário
chamada `ai_model` vencesse o embutido. Implementada assim na 3.3, a regra
mostrou o seu alcance: funções de topo são globais, então **num programa que
define `ai_model`, a sintaxe nativa some em todos os pontos** — inclusive se a
definição vier de uma biblioteca em `use`. O próprio `native-syntax.rob`
definia a função do 7a, e compilado com o robcmp da 3.3 **nenhuma** das 9
chamadas `ai_model("...", N)` dele era reconhecida: todas viravam chamadas à
função do 7a. O arquivo nunca poderia compilar por inteiro, como a 3.9 exige.

**A decisão (orientação).** `ai_model` passa a ser nome reservado: declarar
função, método, tipo, campo, variável (local ou global), parâmetro, interface
ou enum com ele é erro de compilação (`ai_model is a reserved name: it declares
a TinyML model.`). O orientador sugeriu fazer pela lista de palavras reservadas
do léxico (`src/Language.l`); foi feito na análise semântica, com o mesmo efeito
para quem escreve código, porque:

- no léxico, `ai_model(...)` deixaria de ser chamada de função para o parser e
  exigiria uma produção de gramática nova — desfazendo o resultado de 5.2.12;
- o erro seria de sintaxe genérico (`unexpected TOK_AI_MODEL`), e não um que
  diz o motivo.

A checagem fica em `AiModel::checkNotReserved`, chamada no ponto em que todo
nome declarado vira símbolo (`SymbolizeTree::visit(Node&)`,
`src/semantic/SymbolizeTree.h`). Parâmetros não são filhos da função na AST, e
são conferidos ali mesmo, a partir da função.

**Só `ai_model`.** Reservar os nomes dos membros não é viável nem necessário:

- `m.input0.scale` é **um** token só, identificador pontilhado
  (`{ID}(\.{ID})+`, `src/Language.l:111`): uma palavra reservada `scale` nem o
  enxergaria;
- `size` já é membro de vetores e strings (15 arquivos `.rob` do repositório);
- `inputN` é uma família infinita;
- seria recriar o defeito do TCC que a 5.2 manda corrigir — tokens
  `.input`/`.output` colidindo com atributos do usuário (5.2.3).

Os membros são resolvidos pelo valor e não colidem: o 7b da especificação (tipo
do usuário com `input0`, `input()` e `output()`) segue compilando e rodando.

**Custo:** quebra programas que já usassem o nome. Nenhum `.rob` do repositório
o usa, e o nome foi criado para esta extensão. O IR dos 65 programas de
`test/general` é idêntico byte a byte.

**Serve para:** a 5.2, ao descrever a retrocompatibilidade: uma palavra
reservada nova é a mudança clássica e mínima de evolução de linguagem, aqui
restrita a um nome e sem tocar na gramática. E como registro de método: a
especificação executável expôs a contradição antes de ela chegar ao usuário.

### 5.2.14 Nós criados na análise semântica herdavam o arquivo errado

Cada `SourceLocation` guarda o número do arquivo que estava sendo lido quando
foi construído (`src/SourceLocation.h`, `buildStackCurrent`). Um nó criado
**depois** do parse, como o `AiModel` que substitui o `FunctionCall`, ficava com
o arquivo principal. Numa chamada dentro de uma biblioteca compilada com
`-bdep`, o modelo era encontrado no lugar certo (a pasta da biblioteca, porque o
caminho usa o arquivo do `FunctionCall`), mas o erro saía com o nome do `.rob`
principal e a linha/coluna da biblioteca:

```
antes:  …/usa.rob:2:7: semantic error: model '…/lib/sub/m.tflite' ...
depois: …/lib/sub/carrega.rob:2:7: semantic error: model '…/lib/sub/m.tflite' ...
```

Corrigido com `SourceLocation::setFileOf`, que copia o arquivo do nó
substituído. O `ConstructorCall` criado no mesmo ponto
(`PropagateTypes.cpp:354`) tem o mesmo mecanismo; não foi alterado, porque o
erro não foi observado nele e o caso está fora do escopo de TinyML. O teste
`chamada em arquivo de use` de `check_ai_model.py` falha sem a correção
(verificado removendo-a).

**Serve para:** a 5.2 e a 5.4, como exemplo de defeito que só o teste da
mensagem de erro — arquivo, linha e coluna — pega: o comportamento (achar o
modelo) estava certo.

### 5.2.15 Membros do modelo resolvidos contra o `.tflite`, não como símbolos

Os membros de um modelo (`input0`, `output0.raw`, `input0.scale`,
`input_count`, `invoke`…) não são registrados como símbolos. Um resolvedor
único, `ModelMember::resolve` (`src/tinyml/ModelMember.cpp`), interpreta o nome
pontilhado contra o modelo lido: acha o trecho inicial do nome que guarda um
modelo (`m`, `self.m`, `r.m`) e lê o resto contra o `.tflite`. Três efeitos:

- **O erro diz o que existe.** Em vez de `Symbol d.input1 not found`:
  `Model d has 1 input tensor (input0): input1 does not exist. Tensors are
  numbered from 0.`; em vez de símbolo inexistente, `Tensor input0 of model d
  is not quantized: it has no scale.` Cada regra de membro tem uma mensagem
  própria; 26 casos de erro testados.
- **Propriedades viram literais.** `m.input0.size`, `.scale`, `.zero_point`,
  `input_count` e `output_count` são trocados por um nó literal na análise
  semântica; valem até onde a RL exige constante. O teste usa isso: como
  arena, `q.input0.zero_point` dá `must be positive, not -128` — o valor do
  `hello_world_int8.tflite`, lido em tempo de compilação.
- **Nenhuma produção de gramática.** Os nós que o parser já gera (`Load`,
  `LoadArray`, `UpdateArray`, `Scalar`, `Array`, `FunctionCall`) são trocados no
  `PropagateTypes` por um nó por operação: `TensorRead`, `TensorWrite`,
  `ModelInvoke` (`src/tinyml/`). A mudança fora de `src/tinyml` são 36 linhas:
  um gancho de duas a quatro linhas em cada visita, e um `accept` em
  `LoadArray`, que não despachava para a própria visita (nenhum visitor a
  tratava; o comportamento dos demais não muda).

Evidência: `native-syntax.rob` inteiro passa pela análise semântica — os únicos
erros são o aviso de geração não implementada, um por modelo, e o modelo
hipotético de várias entradas da seção 5, que não existe em `models/`. O LLVM
IR dos 65 programas de `test/general` é idêntico byte a byte.

**Serve para:** a 5.2, como o mecanismo que torna a extensão minimamente
intrusiva na semântica também, não só na gramática; e como exemplo de
verificação deslocada para tempo de compilação, com a mensagem como parte do
resultado.

### 5.2.16 A especificação executável pegou um exemplo que violava a própria regra

A seção 4b de `native-syntax.rob` diz que a visão `.raw` exige o tipo exato do
tensor. O exemplo da imagem do person_detect fazia `imagem = {0x0b:9216};
detector.input0.raw = imagem;` — e o compilador da 3.4 o recusou: `The raw view
detector.input0.raw takes int8[], the type of the tensor, not uint8[].` Na RL,
literal hexadecimal é sem sinal (`0x0b` é `uint8`). O exemplo, escrito antes do
compilador, estava errado pela regra que o cercava; corrigido para
`{int8(0x0b):9216}`.

**Serve para:** a metodologia — a especificação como arquivo que o compilador
lê, e não como texto, encontra a contradição no primeiro uso. É o segundo caso
(o primeiro foi a 5.2.13).

### 5.2.17 Como a semântica do TinyML entra no compilador: quatro alternativas comparadas

Depois da implementação dos membros, um teste mostrou um defeito de ordem: uma
função que usa um modelo **global declarado mais abaixo** no arquivo falhava
com erros sem sentido (`Symbol ai_model not found`), enquanto uma global comum
na mesma posição compila. Causa: o `PropagateTypes` percorre o arquivo em
ordem, e o modelo só existia quando a visita chegava à declaração. A correção
abriu a pergunta de onde a semântica do TinyML deve morar. Quatro alternativas,
comparadas antes de escolher (2026-10-14):

| | Linhas no núcleo | Cita TinyML no núcleo | Corrige a ordem | TinyML numa fase só, com tipos |
|---|---|---|---|---|
| **A** ganchos no `PropagateTypes` + declaração sob demanda | 36 (já existentes) | sim, 7 pontos | sim | sim |
| **B** fases próprias, antes e depois do `PropagateTypes` | 2 (`Program.cpp`) | só as chamadas | sim | não |
| **C** ponto de extensão genérico no `PropagateTypes` | ~15 | não | só com A | sim |
| **D** membros como símbolos, tratados em `Load`, `Scalar`... | muitas, espalhadas | sim, em muitos arquivos | sim | sim |

Escolhida a **A**. A **B** parecia a menos intrusiva em linhas, mas o
`PropagateTypes` é onde os tipos passam a existir, e o TinyML depende de tipo o
tempo todo (`float[]` contra `int8[]`, conversão de valores, `.raw`): uma fase
antes dele não sabe tipos, e a semântica do TinyML ficaria dividida em duas.
Além disso, ela teria de percorrer por conta própria todo lugar onde há
expressão — inclusive os elementos de vetor literal, que não são filhos do nó —,
duplicando conhecimento do compilador de forma silenciosa. A **C** cria um
mecanismo de extensão para um único cliente; a **D** espalha o TinyML pelo
núcleo, inclusive pela geração de código, como o compilador faz com
registradores.

A **A** usa o mesmo lugar e o mesmo mecanismo com que o compilador já troca
`FunctionCall` por `ConstructorCall`/`ForcedCast` (5.2.2), e os pontos de
entrada ficam explícitos. A ordem é corrigida dentro de `src/tinyml`, sem
nenhuma linha nova no núcleo: quando o resolvedor encontra um uso cuja
declaração ainda não foi visitada, um modelo global é visitado naquele momento
(as globais existem antes de qualquer função rodar, em qualquer ordem); um
local usado antes da sua declaração é erro, como seria qualquer local.

**Serve para:** a 5.2, como justificativa da arquitetura da extensão — por que a
semântica do TinyML está dentro da propagação de tipos e não numa fase à
parte — e como registro de que a alternativa de menor número de linhas não era a
menos intrusiva.

### 5.2.18 Atribuir um vetor a outro já existente grava um ponteiro dentro dele

Defeito da RL, não de TinyML, encontrado ao decidir o que `saida = m.output0`
faz quando `saida` já existe. Com dois vetores comuns:

```
a = {1, 2};          %a = alloca [2 x i8]
b = {5, 6};          %b = alloca [2 x i8]
b = a;               store ptr %a, ptr %b      <- 8 bytes num espaço de 2
```

Não há cópia nem erro: o endereço de `a` (8 bytes no x86_64) é escrito sobre o
conteúdo de `b` (2 bytes) — escrita fora dos limites na pilha. O programa de
teste devolve `b[0] == 9` depois de `a[0] = 9`, o que parece aliasing, mas o
comportamento é indefinido. Acontece com tamanhos iguais e diferentes. Não
investigado além disso; fora do escopo de TinyML.

**Serve para:** a 5.2, como restrição que a extensão precisa contornar — a
leitura de um tensor inteiro para uma variável existente não pode se apoiar na
atribuição de vetores da RL (decisão pendente, abaixo).

### 5.2.19 `or` entre duas comparações constantes derruba o robcmp

Defeito da RL, não de TinyML, encontrado ao portar os testes para a sintaxe
nativa — as propriedades do modelo são constantes, e o teste as comparava num
`or`. O mínimo que reproduz, em RL pura:

```
if 1 != 1 or 2 != 2 { ... }      Error: signal 11
```

`BinaryOp::logical_operator` trata o valor do primeiro lado como instrução e
pede o bloco dela (`getParent`); com os dois lados constantes, o LLVM já dobrou
a comparação numa constante, que não tem bloco. Contornado nos testes com um
`if` por condição. Não corrigido: fora do escopo de TinyML (decisão de 2026-10-19).

**Serve para:** a 5.2, como limitação da base registrada — e como exemplo de
que constantes de compilação vindas do `.tflite` exercitam caminhos do
compilador que o código RL usual não exercita.

### 5.2.20 O literal `-128` é `int16`

Também da RL. `-128` é lido como `-(128)`, e `128` não cabe em `int8`: o
literal é `int16`. Num vetor, um único `-128` faz o vetor inteiro `int16[]`:

```
a = {11, -72};      [2 x i8]
a = {-128, 127};    [2 x i16]
```

Apareceu ao embutir as imagens do person_detect como `int8[]` para a visão
`.raw`: a imagem tem o byte 0x80. O `rob_embed.py` escreve `int8(-128)` nesse
caso (sufixo `:int8`). Com a visão `.raw` tipada, o engano é erro de compilação
(`takes int8[], the type of the tensor, not int16[]`); no modo biblioteca o
vetor `int16[]` seria entregue a `tflm_set_raw_int8` sem aviso (5.4.9).

### 5.2.21 `__globals_init` rodava depois do código do `main` — corrigido

Defeito da base, corrigido nesta etapa porque a sintaxe nativa depende dele: um
modelo global é criado em `__globals_init`, que o `Program::generate` chama no
início do `main`. A chamada era posta antes da última instrução do bloco de
entrada, contando que ele fosse o bloco das `alloca`, terminado pelo `br` para o
corpo. Quando o `main` não tem variável local, esse bloco é removido
(`FunctionImpl.cpp`), o corpo vira a entrada, e a inicialização ia parar depois
do código. Em RL pura:

```
type contador { n = 7; }
c = contador();
int8 main() { return le(); }      call @le()  e só depois  call @__globals_init()
```

No teste nativo do micro speech, o `main` sem locais classificava o primeiro
áudio com o modelo ainda não criado. Correção (`Program.cpp`, uma linha): a
chamada vai depois das `alloca` (`getFirstNonPHIOrDbgOrAlloca`) — o mesmo
ponto de antes quando há locais, e o início do corpo quando não há. IR de
`test/general` idêntico ao do commit anterior nos 65 programas.

**Serve para:** a 5.2, como a única mudança no núcleo do compilador feita pela
geração de código do TinyML, e por quê.

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

### 5.4.20 Sintaxe nativa: os testes do TCC passam, com valores idênticos

Os testes do TCC reescritos na sintaxe nativa (`test/tflm/rob/nativo_*.rob`),
com as mesmas entradas, limiares e códigos de retorno. O compilador lê o
`.tflite`: só os dados de teste são embutidos.

| Teste | Resultado |
|---|---|
| `nativo_hello_world_float` | OK |
| `nativo_hello_world_int8` | OK |
| `nativo_spam` | OK |
| `nativo_person_detect` | OK |
| `nativo_micro_speech` | OK |

Os testes de índice de tensor inválido (`tflm_input(h, 99)`) não têm
equivalente: `m.input99` é erro de compilação, coberto pelo `check_ai_model.py`.

**Equivalência de valores.** Como na 5.4.19, passar contra limiar não prova
igualdade. Verificado à parte: as 24 saídas dos 11 casos (hello_world float e
int8 com x = 0,0 e 1,57; as duas frases do spam; as duas imagens; os três
áudios) impressas por um programa em sintaxe nativa e por um no modo
biblioteca: **24 de 24 idênticas**. Cada `.raw`, leitura e escrita de vetor e
de elemento passa pelo mesmo wrapper que o modo biblioteca chama.

Reproduzir: `make -C test/tflm nativo`.

### 5.4.21 O modelo fica na flash na sintaxe nativa, e na RAM no modo biblioteca

O mesmo programa (hello_world float: escreve, invoca, lê) nos dois modos,
x86_64, `objdump -h`:

| Seção | Modo biblioteca | Sintaxe nativa |
|---|---|---|
| modelo (3164 bytes) | `.data.modelo` | `.rodata.m.tflite` |
| arena (3000 bytes) | `.bss.arena` | `.bss.m.arena` |
| handle | (local) | `.bss.m.handle` + `.bss.m`, 16 bytes |

No modo biblioteca o modelo é um vetor global da RL, e a RL põe todo vetor
global em `.data` — inclusive declarado `const`, verificado. Num
microcontrolador, `.data` ocupa a flash **e** a RAM (é copiada na
inicialização). Na sintaxe nativa o compilador emite o modelo como constante, e
ele fica só na flash.

O tamanho total quase não muda (+232 bytes na nativa: a função que cria o
intérprete, e o handle guardado). O que muda é onde: no `person_detect`, o modo
biblioteca tem 322400 bytes em `.data` (modelo de ~300 KB mais as imagens); a
sintaxe nativa, 21824 (só as imagens). O STM32F407 tem 192 KB de RAM: **o
person_detect no modo biblioteca não cabe na RAM do alvo**, e na sintaxe nativa
cabe.

**Serve para:** a 5.4 e a Etapa 4 — é a primeira diferença medida entre os dois
modos que não é de conveniência, e decide se o maior modelo de validação roda
no STM32 pelo modo biblioteca. Medir no alvo, com o linker script do
STM32F407.

### 5.4.22 A visão `.raw` tipada recusou o vetor que o modo biblioteca aceitava

O `rob_embed.py` embute bytes em hexadecimal, e a RL infere `uint8[]` (não
`int8[]`, como dizia a documentação do script). O `tcc_person_detect` passa
esse `uint8[]` a `tflm_set_raw_int8(uint64, int8[])` e a RL não reclama — os
bits coincidem, e o teste passa. Na sintaxe nativa, a mesma imagem escrita em
`detector.input0.raw` é erro de compilação:
`The raw view p.input0.raw takes int8[], the type of the tensor, not uint8[].`

**Serve para:** a 5.2, como exemplo concreto do argumento de mover verificação
para a compilação — o tipo do tensor vem do `.tflite`, e o modo biblioteca não
tem como conferi-lo.

### 5.4.23 Quatro modelos no máximo, e o quinto falha calado

O wrapper tem `TFLM_MAX_MODELS` = 4 instâncias. Na sintaxe nativa não há
destruição (seção 8b da especificação), então cada declaração `ai_model(...)`
ocupa uma instância para sempre. Um programa com 5 modelos compila, e o quinto
falha na criação: a variável guarda 0, `invoke()` devolve -1 e as leituras dão
0. Medido: o programa de equivalência da 5.4.20 com os cinco modelos juntos
devolveu zeros no micro speech.

O compilador conhece todas as declarações: o limite pode virar erro de
compilação. Em aberto (abaixo): o número é um `#define` do wrapper, compilado
à parte.

**Resolvido em 2026-10-19.** Primeiro implementado como erro em toda
declaração além da quarta — e a própria especificação deixou de compilar: o
`native-syntax.rob` tem um exemplo por função, cada um com o seu modelo, e
nenhum par deles roda junto. A regra "toda declaração conta" recusaria também
um firmware real que escolhe o modelo pelo modo de operação. A versão final
separa o que o compilador sabe do que não sabe: modelos **globais** são todos
criados antes do `main`, então mais deles que instâncias é **erro**; um modelo
**local** só é criado se a função rodar, então passar do limite com locais é
**aviso**, na primeira declaração além do limite. Verificado na geração de
código, quando todas as declarações já foram vistas (um global pode vir depois
das funções que o usam). O limite está no compilador
(`src/tinyml/AiModel.cpp`); o `check_ai_model.py` lê o `TFLM_MAX_MODELS` do
wrapper, e um número diferente do outro faz o teste falhar.

**Serve para:** a 5.2, como exemplo de verificação que só pode ser parcial —
o compilador transforma em erro o que é certo e em aviso o que depende da
execução, em vez de recusar programas válidos.

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

### 2026-09-28 — Etapa 3.1: módulo de análise de `.tflite` no compilador

`src/tinyml/TfliteModel.{h,cpp}`, compilado dentro do robcmp. O
`test/tflm/tflite_info.cpp` passou de implementação própria a executável fino
sobre o módulo, e deixou de compilar o `schema_utils.cc` do TFLM: o
`GetBuiltinCode` foi reimplementado (uma linha, `max` dos dois campos).

Acrescentados ao que o `tflite_info` fazia: verificação estrutural (5.2.9),
kernels pelos operadores (5.2.10), a função do wrapper de cada kernel, e erros
por tensor inexistente, shape dinâmico e custom op sem nome.

`make -C test/tflm check` ganhou o alvo `tflite` (`check_tflite.py`): tabela ×
header (119 = 119), fatos dos 7 modelos de `models/` — o teste falha se entrar um modelo sem fatos esperados — (incluindo o `spam` da 5.2.4 e os 9
custom ops do áudio) e robustez sob AddressSanitizer. Suíte completa passa.

### 2026-10-05 — `4dd5c68` Etapa 3.2: produções da gramática (revertido em 2026-10-13)

As formas P1, P2 e P3 entraram em `src/Language.y`, gerando nós que só
registram a forma sintática (`src/ElementAccess.h`): `UpdateElementArray`,
`LoadElementMember`, `UpdateElementMember`. O parser não conhece o tipo do
objeto indexado; a semântica dá o significado. Até o `ai_model` existir (3.3),
todo uso é um único erro semântico, na linha e coluna certas
(`Member access a[...].scale is only valid on ai_model tensors.`).

Evidência de extensão mínima: o Bison continua com o 1 conflito já declarado
com `%expect 1`, com o mesmo contraexemplo (`'(' expr • ')'`); o LLVM IR dos 65
programas de `test/general` é idêntico byte a byte antes e depois; o placar dos
testes não muda (63 passam; `bool.rob` e `scalar-in-two-contexts.rob` já
falhavam).

Notado para a 3.4 (superado em 2026-10-13: no desenho com membros o elemento é
`m.output0[5]`, um `LoadArray`, e não há duplo índice): `m.output[0][5]` e
`m.input[0][7] = v` já eram aceitos pela gramática, como `LoadMatrix`/`UpdateMatrix`. Escrita com
duplo índice num símbolo inexistente aborta o robcmp
(`Assertion 'tid != -1 && "Undefined type"'`), também na base.

### 2026-10-06 — Demonstrações executáveis dos achados (`test/achados`)

Pasta com um alvo de Makefile por achado, para mostrar na orientação:
`make signext` (5.2.7), `make use` (5.2.11), `make arena` (5.2.8). Construí-las
corrigiu dois registros: a 5.2.7 descrevia o mecanismo errado (não há `zeroext`
em `int8`; falta `signext`, e o valor sai de fato errado), e a 5.2.8 não dizia o
alinhamento do buffer em que foi medida.

### 2026-10-13 — Sintaxe com tensores como membros; P1–P3 revertidas

Aplicadas à especificação as decisões da orientação de 2026-10-05 (5.2.12):
`inputN`/`outputN`, `input_count`/`output_count`, caminho relativo ao `.rob` e
depois `-I`. Numeração trocada de 1 para 0 (5.2.12). Acrescentados à seção 8 os
erros de tensor além do último e de índice de elemento constante fora do
tensor. Corrigidas na especificação a API citada
no modo biblioteca (ainda dizia `InitializeInterpreter`) e a origem dos kernels
(5.2.10). O commit `4dd5c68` (P1–P3) foi revertido: a gramática é a original.

### 2026-10-13 — Etapa 3.3: `ai_model(...)` na análise semântica

Novo nó `AiModel` (`src/tinyml/AiModel.{h,cpp}`), a declaração do modelo — o
primeiro dos módulos que substituem o `ModelNode` do TCC. Em
`PropagateTypes::visit(FunctionCall)` (`PropagateTypes.cpp:324`), toda chamada a
`ai_model` é trocada por ele; o nome é reservado (5.2.13). O nó:

- confere a aridade (arena ausente tem mensagem própria), o caminho literal
  (`StringConst`, que ganhou `getString()`) e a arena: constante inteira
  positiva — literal, `const` nomeada ou `+ - *` sobre elas. O `-5` precisou
  disso: o parser o constrói como `5 * -1` (`Language.y:462`);
- procura o arquivo na pasta do `.rob` da chamada e depois nas `-I`, na ordem
  dada. As `-I` passaram a ser guardadas também à parte (`cmdlineIncludeDirs`,
  `src/Scanner.cpp`), porque `includeDirs` as mistura com as pastas dos fontes e
  é percorrida de trás para frente. A pasta de trabalho não entra;
- lê o modelo com `TfliteModel::load` e o guarda no nó (`getModel()`), com a
  arena (`getArenaSize()`).

Todo erro substitui a chamada mesmo assim, por um nó marcado inválido, para que
ela não seja reportada de novo como função inexistente. Um modelo aceito para na
geração de código com `model '...' was read, but the code generation of ai_model
is not implemented yet.` — a fronteira até a 3.6.

Teste novo, `make -C test/tflm ai_model` (`check_ai_model.py`), dentro do
`check`: 33 casos — os 11 erros de declaração da seção 8 e 3 de arena aceita,
em arquivo, linha e coluna; a ordem de busca (7 casos, inclusive `-I`
invertidas e chamada em arquivo de `use`); o nome reservado em 11 formas de
declaração; e o 7b (tipo do usuário com `input0`, `input()` e `output()`,
compilado, ligado e executado). O LLVM IR dos 65 programas de
`test/general` é idêntico byte a byte antes e depois; `make -C test/tflm check`
passa por inteiro.

Achados: 5.2.13 (a precedência da função do usuário desligava a sintaxe no
programa inteiro; trocada pela reserva do nome) e 5.2.14 (arquivo errado nos
nós criados na semântica, corrigido). Especificação atualizada (7a vira nome
reservado, e o caso entra na seção 8). `lib/ai/tflm.rob`: a tabela de
correspondência com a sintaxe nativa ainda era do desenho indexado
(`m.input[i]`, `m.output[i][j]`); reescrita com os membros e completada com
criação, `invoke`, contadores e liberação.

Notado, pré-existente (do upstream, `d957142`): um erro semântico reportado na
**primeira** declaração do arquivo sai na posição errada. São dois defeitos
somados:

1. `update_loc` (`src/Language.l:165`) faz cada token começar onde o anterior
   terminou; o primeiro token do arquivo não tem anterior e fica com `0:0`.
2. A regra `globals : global` (`src/Language.y:62`) sobrescreve a posição da
   primeira declaração (a do nome, `@id`) com `@global`, o início da declaração
   inteira — o tipo, que é o primeiro token. A regra `globals : globals global`
   não faz isso, por isso só a primeira é afetada.

Com a mesma declaração `int16 ai_model(...)`: o arquivo começando direto no
código dá `0:0`; começando com linha em branco ou comentário, dá `2:1` (o
`int16`, não o nome); o certo seria `2:7`. Corrigir só o léxico não basta. Não
corrigido — fora do escopo de TinyML; os casos de nome reservado do teste põem
uma declaração antes.

Notado, pré-existente: escrever num elemento de símbolo inexistente aborta o
robcmp (`Assertion 'tid != -1 && "Undefined type"'` em `BuildTypes::isEnum`),
também com índice simples — `nada[7] = 0.42;` num `main` vazio basta. O caminho
é `PropagateTypes::visit(UpdateArray)` (`PropagateTypes.cpp:429`), que coage o
valor para o tipo do elemento sem conferir se ele existe. A nota da 3.2 só
registrava o duplo índice.

### 2026-10-13 — Etapa 3.4: membros do modelo na análise semântica

Antes do código, quatro lacunas da especificação, fechadas em
`native-syntax.rob` (decididas ao começar a tarefa):

- tensor sem quantização não tem `.scale` nem `.zero_point`: ler é erro (seção 3);
- a visão padrão é sempre `float`, para todo tipo de tensor; a `.raw` tem o tipo
  do tensor e só existe para `int8` e `int16`, os tipos com `tflm_*_raw_*` no
  wrapper (4b);
- tensores de saída são somente leitura; o modelo não é um valor — `x = m`,
  `f(m)`, declará-lo de novo, atribuir a ele, e `ai_model(...)` fora de
  `m = ai_model(...)` são erro (4c, nova).

Implementação em `src/tinyml/`: `ModelMember` (o resolvedor, 5.2.15),
`ModelUse` (as seis trocas, chamadas do `PropagateTypes`), e um nó por
operação — `TensorRead`, `TensorWrite`, `ModelInvoke` —, como a seção 5.2 do
`CLAUDE.md` pede no lugar do `ModelNode`. Tipos: o tensor inteiro é `float[]`
(ou `int8[]`/`int16[]` na `.raw`), com `.size`, e o elemento é escalar;
`invoke()` é `int8`. O modelo em si tem o tipo `uint64`, o handle do
intérprete, como no modo biblioteca — nunca visível como número, porque o modelo
não é valor.

Duas mudanças no `AiModel` por causa da geração ainda ausente:

- o aviso "code generation ... not implemented yet" passou da geração para a
  análise semântica. Na geração, os nós novos não devolvem valor, e
  `return m.invoke()` derrubava o robcmp (conversão `int8`→`int16` sobre nulo).
  Reportado na semântica, o erro impede a geração de rodar. Mesma mensagem,
  mesma posição; a 3.6 o remove;
- modelo como campo de tipo é recusado com mensagem própria (decisão em aberto,
  abaixo). Antes, dava `The type of m is undefined`.

`lib/ai/tflm.rob` já tinha sido corrigido na 3.3; o mesmo comentário com o
desenho indexado (`m.input[i]`) estava no wrapper (`tflm_wrapper.{h,cpp}`) e foi
reescrito com os membros. Só comentários.

Teste: `check_ai_model.py` ganhou a etapa de membros — 36 casos (69 no total):
a especificação inteira pela semântica, 4 constantes com o valor do `.tflite`, 4
tipos de visão expostos pela mensagem de tipo errado, 26 erros em linha e
coluna e o modelo como campo de tipo. A regra "saída é somente leitura",
retirada à mão, faz 3 casos falharem. IR de `test/general` idêntico;
`make -C test/tflm check` passa.

Fica para depois: tamanho do vetor contra o do tensor e índice constante fora
do tensor (3.5, 3.7) — `d.input0 = {1.0, 2.0}` ainda passa; kernel sem suporte
(3.7).

### 2026-10-14 — Ordem das declarações de modelo; arquitetura revista

Defeito encontrado ao revisar a integração: modelo global declarado depois da
função que o usa não compilava. Comparadas quatro formas de integrar a
semântica do TinyML ao compilador (5.2.17); mantida a dos ganchos no
`PropagateTypes`, com a declaração resolvida sob demanda:
`AiModel::isPendingDeclaration` reconhece a declaração ainda não visitada, e
`ModelMember::resolve` (que agora recebe o `PropagateTypes`) a visita na hora,
se for global, ou dá `m is used before its declaration`, se for local. Uma
declaração já visitada assim não é processada de novo quando a visita normal
chega a ela.

Regra registrada na seção 4c da especificação (onde o modelo vale), e o caso do
local na seção 8. Teste: 3 casos novos (72 no total) — global depois de duas
funções que o usam, lido uma vez só; erro de membro nesse global, reportado
uma vez; local usado antes. Retirar a visita sob demanda faz os dois primeiros
falharem. IR de `test/general` idêntico ao do commit anterior;
`make -C test/tflm check` passa.

### 2026-10-14 — Etapa 3.5: tamanhos e índices em tempo de compilação

Nas duas visões, não só na `.raw`: o mecanismo é o mesmo, e a seção 8 da
especificação já trazia o caso da visão padrão (`d.input0 = {1.0, 2.0}`). A
seção 4b passou a dizer isso explicitamente.

- **Vetor escrito no tensor inteiro:** quando o tamanho é conhecido na
  compilação — literal, variável que guarda um, outro tensor lido inteiro —
  precisa ser o do tensor: `d.input0 has 1 element; the vector has 2.` Vetor de
  tamanho desconhecido (parâmetro `float[]`) fica para o wrapper.
- **Índice constante de elemento:** de 0 a `size - 1`, inclusive constante
  nomeada e negativo: `m4.output0 has 4 elements: index 5 is out of the tensor.`
  Índice variável fica para o wrapper.

O cálculo de constante inteira saiu do `AiModel.cpp` para
`src/tinyml/IntegerConstant.{h,cpp}`, junto com `staticSize`, que lê o `.size`
constante de um vetor — o mesmo `NamedConst` que literais, variáveis que os
guardam e `TensorRead` já expõem.

- **Tensor lido para variável que já existe** (decidido ao fim da tarefa): é
  copiado para ela — o buffer reservado uma vez e reaproveitado a cada
  inferência. A variável precisa ter o tipo da visão (`q.output0.raw is int8[],
  and c, which receives it, is int16[].`) e, quando conhecido, o tamanho do
  tensor (`sp.output0 has 2 elements, and saida, which receives it, has 3.`).
  Não se apoia na atribuição de vetores da RL, que grava um ponteiro dentro do
  vetor de destino (5.2.18). Regra e exemplos na seção 4b e na seção 8.

Teste: etapa nova no `check_ai_model.py`, 20 casos (92 no total) — 12 erros em
linha e coluna e 8 casos que precisam passar (tamanho certo, último elemento,
índice variável, tensor lido de mesmo tamanho, parâmetro de tamanho
desconhecido, buffer reservado, buffer relido de outro modelo, leitura no
laço). IR de `test/general` idêntico ao do commit anterior;
`make -C test/tflm check` passa.

### 2026-10-18 — Etapa 3.6: geração de código; 3.7 concluída

A sintaxe nativa compila e roda. Para `m = ai_model("modelo.tflite", 3000)`, o
compilador emite (`src/tinyml/AiModel.cpp`):

- `@m.tflite`, os bytes do modelo, constante; `@m.arena`, zerada; os dois
  alinhados a 16;
- uma função `m.ai_model()` que cria o intérprete nas três fases do wrapper,
  com um `tflm_add_*` para cada kernel do `.tflite`, e guarda o handle. A
  declaração a chama. O modelo é criado uma vez (seção 4c): uma declaração
  local, numa função que roda de novo, reaproveita o intérprete — sem isso, a
  quinta chamada esgotaria as instâncias do wrapper. Criação que falha libera a
  instância e deixa 0 na variável;
- um modelo global é criado em `__globals_init`, antes do `main`.

Leitura, escrita e `invoke()` (`TensorRead.cpp`, `TensorWrite.cpp`,
`ModelInvoke.cpp`) viram chamadas da API `tflm_`, declaradas por
`src/tinyml/WrapperCall` com as mesmas assinaturas do modo biblioteca — um
programa que usa os dois modos compartilha as declarações. Tensor lido inteiro
para variável nova ganha um buffer próprio; para variável que já existe, é
copiado para o vetor dela (5.2.18).

`.raw[k]` precisou de função nova no wrapper, `tflm_set_raw_value` e
`tflm_get_raw_value`, para tensor int8 ou int16. O código trafega como `int32`
nos dois sentidos: um `int8` escalar dependeria de extensão de sinal na
chamada, que o robcmp não marca (5.2.7). Declaradas também em `lib/ai/tflm.rob`;
o `check_abi.py` confere as duas (139 funções).

3.7: falta só o kernel sem suporte, agora erro na declaração (`model '...'
uses the builtin kernel LSH_PROJECTION, which the TFLM wrapper does not
support.`), testado com um `.tflite` cujo operador foi trocado.

Testes: `make -C test/tflm nativo` (camada 3, 5.4.20), incluída no `check`.
`check_ai_model.py` deixa de depender do erro provisório "not implemented
yet": um programa aceito precisa compilar sem erro — o que inclui o
verificador do LLVM sobre o IR gerado —, e "lido uma vez" se confere contando
os `@<variável>.tflite` do IR. 94 casos. Uma correção no núcleo (5.2.21); IR
de `test/general` idêntico ao do commit anterior.

Fora do recorte, ficam para a sequência da 3.6:

- **Quantização com constantes** (seção 2 da especificação): hoje a conversão
  float ↔ int8 é a do wrapper, em tempo de execução, com `scale`/`zero_point`
  lidos do tensor. O resultado é o mesmo (5.4.20); a especificação prometia a
  conversão emitida pelo compilador, sem divisão em runtime. Decidido depois,
  no mesmo dia: fica no wrapper (ver as decisões).
- **Limite de modelos** em tempo de compilação (5.4.23) — feito em 2026-10-19.

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

### ⬜ Etapa 3 — Compilador, seção 5.2 (EM ANDAMENTO)

Pela regra "sintaxe antes da implementação" (`CLAUDE.md`, seção 6), a primeira
tarefa é fechar a especificação. As demais seguem a ordem das fases do
compilador: elaboração prévia, gramática, semântica, geração de código.

| # | Tarefa |
|---|---|
| 3.0 | Fechar as decisões de `test/tflm/native-syntax.rob` | **feito** — decisões da orientação de 2026-10-05 aplicadas (5.2.12); nome `ai_model` confirmado |
| 3.1 | Migrar a leitura do `.tflite` para um módulo em `src/`, como fase de elaboração prévia | **feito** — `src/tinyml/TfliteModel` (5.2.9, 5.2.10) |
| 3.2 | Gramática | **dispensada** — o desenho com membros não exige produção (5.2.12); P1–P3 (`4dd5c68`) revertidas |
| 3.3 | `ai_model(...)` na semântica (`PropagateTypes::visit(FunctionCall)`): nome reservado (antes: precedência da função do usuário), caminho literal resolvido relativo ao `.rob` e depois nas pastas `-I`, arena obrigatória, `TfliteModel::load` | **feito** — nó `AiModel`; `make -C test/tflm ai_model` (5.2.13, 5.2.14) |
| 3.4 | Membros do modelo derivados do `.tflite`: `inputN`/`outputN` (vetores tipados), `input_count`/`output_count`, e por tensor `.size`/`.scale`/`.zero_point`/`.raw`. Interceptar os nós que o parser já gera: `Load`, `LoadArray`, `UpdateArray`, `Scalar`, `Array`, e `FunctionCall` para `m.invoke()` | **feito** — `ModelMember`, `ModelUse`, `TensorRead`/`TensorWrite`/`ModelInvoke` (5.2.15, 5.2.16) |
| 3.5 | Visão `.raw`: verificação de tipo e tamanho em tempo de compilação, inclusive `.raw[k]` | **feito** — tipo na 3.4; tamanho e índice nas duas visões; leitura para variável existente |
| 3.6 | Geração de código: modelo e arena como globais alinhados a 16 (5.2.8), um `tflm_add_*` por kernel detectado, e as chamadas da API `tflm_` (5.1.7) | **feito** — os 5 modelos rodam, valores idênticos ao modo biblioteca (5.4.20); conversão float ↔ quantizado mantida no wrapper (decisão) |
| 3.7 | Erros de compilação da seção 8 da especificação: caminho não literal, arena ausente, arquivo inválido, tamanho incompatível, tensor inexistente (como `input1` num modelo de uma entrada), índice de elemento constante fora do tensor, escrita em `.scale`, `.raw` com tipo ou tamanho errado, kernel sem suporte | **feito** — kernel sem suporte na 3.6 |
| 3.8 | `signext` em `int8`/`int16` com sinal, em parâmetros e retornos (5.2.7) | **não será feito** — defeito da base, contornado por regra no wrapper (decisão de 2026-10-19) |
| 3.9 | Fazer `test/tflm/native-syntax.rob` compilar e rodar, com o mesmo resultado do modo biblioteca |

A especificação executável da sintaxe está em `test/tflm/native-syntax.rob`.

**Interface deixada pela 3.1** (`src/tinyml/TfliteModel.h`), para a 3.3 em diante:

- `TfliteModel::load(caminho, modelo, erro)` — lê e verifica o arquivo uma vez.
  Falha devolve `false` e uma mensagem em inglês nomeando o arquivo: `file not
  found or not readable`, `not a TFLite model (TFL3 identifier missing)`,
  `corrupted or truncated TFLite file`, tensor inexistente, shape dinâmico,
  custom op sem nome. O chamador acrescenta a localização no `.rob`.
- `modelo.inputs`/`outputs` — do subgrafo 0: tipo (`TfliteType`), shape,
  `elements` (achatado), `quantized`, `scale`, `zeroPoint`, `quantChannels`.
- `modelo.kernels` — sem duplicatas, ordenados; `wrapperFunction` vazio =
  kernel sem suporte (erro da 3.7, a ser emitido pela semântica).
- `modelo.bytes` — o arquivo inteiro, para a 3.6 embutir no objeto.
- O header não expõe nada do TFLM: só `TfliteModel.cpp` inclui o schema, e só
  ele recebe os include dirs do TFLM no `CMakeLists.txt`, que falha com
  mensagem clara se faltar o submódulo ou o `third_party_downloads`.
- A tabela kernel → `tflm_add_*` fica no `.cpp`; `make -C test/tflm tflite`
  confere que ela cobre exatamente as funções do header e que cada custom op é
  um nome registrado pelo resolver do TFLM.

**Interface deixada pela 3.3** (`src/tinyml/AiModel.h`), para a 3.4 em diante:

- `AiModel::checkNotReserved` — chamada pelo `SymbolizeTree` para todo nome
  declarado; é o único lugar que conhece a reserva.
- `AiModel` substitui o `FunctionCall` em `PropagateTypes`; é um `Constructor`
  (tem `setLeftValue`), sem filhos — os argumentos já foram consumidos.
- `isValid()` — falso se algum erro foi reportado; os passos seguintes devem
  ignorar um nó inválido, para não somar erros em cascata.
- `getModel()` — o `TfliteModel` lido (`inputs`, `outputs`, `kernels`, `bytes`,
  e `path`, o caminho como encontrado); `getArenaSize()` — a arena, já conferida.
- `getDataType()` ainda é `undefinedType`, e `generate()` só reporta que a
  geração não existe. Os dois mudam na 3.4 e na 3.6; o `check_ai_model.py`
  confere essa mensagem nos casos de sucesso e precisa ser ajustado junto.

**Interface deixada pela 3.4** (`src/tinyml/`), para a 3.5 em diante:

- `ModelMember::resolve(nome, escopo)` — o que um nome pontilhado é num modelo:
  `None` (não é de modelo), `Model`, `Tensor`, `Property`, `Invoke` ou `Error`
  (mensagem pronta; vazia se a declaração já tinha erro). O `TensorRef` diz o
  modelo, a variável que o guarda, entrada/saída, índice e visão, e dá
  `elementType()`/`vectorType()`.
- `TensorRead(tensor, elemento)` e `TensorWrite(tensor, elemento, valor)` —
  elemento `NULL` = o tensor inteiro. O valor da escrita inteira é um vetor já
  tipado: um `Array` literal (o próprio nó do parser, ainda com o nome
  `m.input0`) ou qualquer expressão de vetor. `ModelInvoke(modelo)`.
- Os `generate()` dos três devolvem `NULL`, e o do `AiModel` também; nenhum é
  alcançado enquanto o aviso da semântica existir (`AiModel::fromCall`).
  (Implementados na 3.6, e o aviso removido.)
- Leitura para variável que já existe (`saida = m.output0`, com `saida`
  declarada antes): a 3.6 gera a cópia para o buffer existente
  (`tflm_get_values`/`tflm_get_raw_*` sobre ele), e não a atribuição de
  vetores da RL, que é defeituosa (5.2.18). Numa variável nova, o `TensorRead`
  gera o próprio buffer.
- Uma declaração global pode ser visitada antes da sua vez, por um uso acima
  dela no arquivo (5.2.17). Na 3.6, a geração dos globais precisa acontecer
  antes da das funções que os usam — como já acontece com as globais comuns.

**Notas para a 3.5 e a 3.6** (levantadas em 2026-10-13):

- **Tamanhos (3.5/3.7):** o tamanho do literal está no `Array`
  (`getSize()`); o de um vetor em variável, no seu `.size`. O índice constante
  do elemento está em `TensorRead/TensorWrite::getElement()`.
- **Geração (3.6):** tirar o aviso de `AiModel::fromCall` e ajustar o
  `check_ai_model.py`, que o usa como marca de modelo aceito. O `Array` literal
  da escrita precisa ser gerado como temporário — hoje ele se chama `m.input0`.
- **Função chamada várias vezes (3.6):** `ai_model(...)` dentro de uma função
  cria o intérprete a cada chamada, sobre a mesma arena global. Decidir junto
  com o armazenamento de modelo e arena.
- **Seção 5 da especificação (3.9):** o modelo de várias entradas é hipotético.
  Para a 3.9 compilar o arquivo inteiro, usar uma fixture do TFLM com duas
  entradas (o `simple_add_model`, citado na seção 5) ou comentar o exemplo.

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


- Arena automática (fase 2): o caminho é medir na compilação, rodando o TFLM do
  host (`test/achados/arena`). Falta saber se o valor do host cobre o do
  Cortex-M — decide-se com as medições no STM32 da Etapa 4.
- Modelo como campo de tipo (`type robo { m = ai_model(...); }`): fora da
  especificação; recusado com erro próprio. Duas instâncias do tipo dividiriam
  o modelo e a arena — decide-se com o armazenamento, na 3.6.

Decidida em 2026-10-19: limite de modelos como erro para globais e aviso com
locais (5.4.23), com o número no compilador igual ao `TFLM_MAX_MODELS`.

Decidida em 2026-10-19: defeitos da RL base encontrados durante o TinyML são
registrados e contornados, não corrigidos — a falta de `signext` (5.2.7), o
`or` entre comparações constantes (5.2.19) e o literal `-128` (5.2.20). A
extensão de TinyML não depende de nenhum deles: o wrapper não tem parâmetro
`int8`/`int16` escalar, regra agora escrita no `CLAUDE.md`. Candidatos a
reportar ao upstream. A única mudança feita no núcleo segue sendo a do
`__globals_init` (5.2.21).

Decididas na 3.6 (2026-10-18): o modelo é criado uma vez, também quando
declarado numa função que roda de novo; `.raw[k]` por função própria no
wrapper, com o código em `int32`; a conversão float ↔ quantizado fica no
wrapper, em tempo de execução, e não é emitida pelo compilador com `scale` e
`zero_point` constantes (seção 2 da especificação). Motivos: o ganho é por
elemento de entrada e saída, pequeno diante da inferência; multiplicar pelo
inverso da escala não arredonda igual a dividir, e os dois modos deixariam de
dar resultados idênticos (5.4.20); e a regra ficaria em dois lugares.

Confirmado (registrado em 2026-10-16): o nome do construtor é `ai_model`.

Decidida na 3.5 (2026-10-14): tensor lido inteiro para variável que já existe
é copiado para ela; tipo e, quando conhecido, tamanho precisam ser os do tensor.

Decididas ao começar a 3.4 (2026-10-13): tensor sem quantização não tem
`scale`/`zero_point`; visão padrão sempre `float` e `.raw` só para `int8` e
`int16`; saídas somente leitura; o modelo não é um valor.

Decidida em orientação posterior a 2026-10-05: `ai_model` é nome reservado,
na análise semântica (5.2.13) — substitui a precedência da função do usuário.

Decididas na orientação de 2026-10-05: tensores como membros numerados
(5.2.12), a partir de 0 — a base 1 proposta foi trocada depois; `input_count`/`output_count` no modelo; caminho do modelo relativo
ao `.rob` e depois as pastas `-I`; arena manual na primeira versão.

---

## Estado do repositório

Branch `mestrado/tflm-native`. Etapas 1 e 2 commitadas; Etapa 3 em andamento:
3.0, 3.1 e 3.3 a 3.7 concluídas, 3.2 dispensada, 3.8 descartada (defeito da
base); falta a 3.9. A sintaxe nativa compila e roda os cinco
modelos de validação. `make -C test/tflm check` passa por inteiro, incluindo os
alvos `nativo` e `ai_model`; `make -C test/achados` reproduz os achados 5.2.7,
5.2.8 e 5.2.11.

Os dados de teste `test/tflm/data/*.bin` estão versionados desde 2026-10-18: o
`.gitignore` mantém `*.bin` (firmware gerado em `hack/stm32` e pelo PlatformIO)
e abre exceção para `test/tflm/data/*.bin`. Um clone novo roda o `make -C
test/tflm check` inteiro.
