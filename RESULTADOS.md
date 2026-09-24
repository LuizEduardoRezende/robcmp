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


---

## Seção 5.2 — Frontend e análise estática

### 5.2.1 A sintaxe nativa exige apenas duas produções novas de gramática

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

### ⬜ Etapa 2 — Validação (PRÓXIMA)

**Situação crítica a registrar: nada foi executado até aqui.** Toda a
verificação da Etapa 1 foi em nível de compilação e símbolos. A
`libtensorflow-microlite.a` nunca foi compilada e nenhuma inferência rodou.

Ordem sugerida, do menor risco para o maior:

| # | Tarefa | Por quê nesta ordem |
|---|---|---|
| 2.1 | Compilar `libtensorflow-microlite.a` (x86_64) | pré-requisito de tudo |
| 2.2 | Linkar e rodar `hello_world_float` | menor modelo: 1 kernel, 1 elemento de entrada e saída, sem quantização |
| 2.3 | Rodar `hello_world_int8` | primeiro a exercitar a quantização unificada no B6 |
| 2.4 | Rodar `spam` | entrada `int32`, saída `float32` — tipos diferentes nas duas pontas |
| 2.5 | Rodar `person_detect` | 5 kernels, entrada de 9216 elementos |
| 2.6 | Rodar `micro_speech` + `audio_preprocessor` | exercita os 9 custom ops novos e o pipeline de dois estágios |
| 2.7 | Adaptar os 5 `.rob` de teste do TCC à API nova | base da comparação retroativa da 5.4 |

Comando para 2.1:

```bash
make -C third-party/tflite-micro -f tensorflow/lite/micro/tools/make/Makefile microlite
```

O que investigar primeiro se algo falhar: a Etapa 1 trocou construção dinâmica
por *placement new* (`g_instances`), mudou toda a API de ciclo de vida e
reescreveu a conversão de tipos. Suspeitos na ordem: `in_use` não resetado,
alinhamento do `interpreter_storage`, conversão de handle.

---

### ⬜ Etapa 3 — Compilador, seção 5.2

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
