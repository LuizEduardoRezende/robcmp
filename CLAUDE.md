# CLAUDE.md — Robcmp / Robotics Language: Integração Nativa de TinyML

Este arquivo orienta o Claude Code no desenvolvimento da pesquisa de mestrado
(PPGCC-UFJ) que estende o compilador **Robcmp** para suportar a declaração e
execução de modelos TinyML nativamente na sintaxe da **Robotics Language (RL)**.

> Este documento é contexto operacional para o agente de codificação. Ele não
> substitui o projeto de pesquisa nem o TCC — para justificativas teóricas,
> revisão de literatura e redação da dissertação, esses documentos devem ser
> consultados/mantidos separadamente (ex.: Google Docs de sistematização
> mencionado no projeto, seção 6.1.5).

> **Estado do trabalho:** `RESULTADOS.md`, na raiz, registra o que já foi feito,
> os achados com evidência (e para qual seção da dissertação cada um serve), o
> roteiro ordenado do que falta e as decisões em aberto. **Consulte-o no início
> de cada sessão** — este arquivo descreve o projeto, aquele descreve o estado.

---

## 1. Contexto do projeto

- **Repositório de trabalho:** fork de `thborges/robcmp`
  (https://github.com/LuizEduardoRezende/robcmp). O upstream oficial é
  https://github.com/thborges/robcmp/.
- **Branches relevantes:**
  - `tcc/tflm-front-end` — branch do TCC (dez/2025), criada a partir da base `llvm20`. Contém a PoC original (wrapper externo + `ai.tflm`), validada apenas em x86_64. **Não editar diretamente** — é o baseline histórico de comparação/retrocompatibilidade.
  - `mestrado/tflm-native` — branch atual de trabalho do mestrado, criada a partir da base `cola2025`. **É nela que o desenvolvimento desta pesquisa deve ocorrer.**
  - Por conta da base diferente (`llvm20` vs. `cola2025`), a branch `mestrado/tflm-native` **não é um simples avanço linear** sobre `tcc/tflm-front-end`: o frontend do compilador (léxico, sintático, semântico) pode ter tokens, regras de gramática e estrutura de AST distintos entre as duas bases. Qualquer reaproveitamento de código do TCC deve ser adaptado, não copiado diretamente — conforme já previsto na seção 6.1.2 do projeto de pesquisa.
- **Estado anterior (TCC, dez/2025):** prova de conceito de inferência TinyML
  na RL via wrapper externo C/C++, validada apenas em x86_64.
- **Objetivo atual (mestrado, abr/2026–abr/2028):** transformar essa PoC em
  suporte nativo no frontend do compilador (léxico, sintático, semântico),
  refatorar o wrapper, criar um pipeline de build unificado, e validar em
  hardware embarcado real (STM32F407VET6).
- **IMPORTANTE — princípio de design vigente:** qualquer mudança no frontend
  deve ser uma **extensão minimamente intrusiva** da gramática/semântica
  existentes da RL (taxonomia de Mernik/Heering/Sloane) — não é a criação de
  uma nova linguagem, e não deve quebrar a compatibilidade retroativa da
  biblioteca padrão `ai.tflm`.

---

## 2. Stack de tecnologia

| Camada | Ferramenta | Observações |
|---|---|---|
| Linguagem do compilador | C++ | Robcmp é escrito em C++ |
| Análise léxica | Flex 2.6.4 | arquivo(s) `.l` |
| Análise sintática | Bison 3.8.2 | gramática LALR |
| Backend / geração de código | LLVM (IR + backend) | versão mantida atualizada continuamente; suporta x86, ARM (Cortex-M), AVR |
| Linker | `ld.lld` | invocado via script Python do PlatformIO |
| Linguagem-alvo (DSL) | Robotics Language (RL), arquivos `.rob` | tipagem implícita, interfaces, injeção de dependência dirigida pelo compilador |
| Engine de inferência TinyML | TensorFlow Lite Micro (TFLM) | licença Apache 2.0; abordagem por interpretador |
| Parsing de modelo `.tflite` | FlatBuffers (binding C++) | `schema_generated.h`, gerado a partir de `schema.fbs` do TF Lite |
| Ambiente de build embarcado | PlatformIO (v6.1.18+) | orquestra Robcmp + linker + upload |
| IDE recomendada | VS Code | extensões: **PlatformIO** e **RobCmpSyntax** (syntax highlighting para `.rob`) |
| Benchmark nativo (x86_64) | Hyperfine | warmup 10 iterações, 100 medições, métrica `Time (mean)` |
| Benchmark embarcado | DWT (Data Watchpoint and Trace) | periférico nativo Cortex-M4, contador de ciclos de 32 bits, exposto à RL como registrador mapeado em memória |
| MCU alvo (novo) | STM32F407VET6 (ARM Cortex-M4) | ainda não suportado; requer novo linker script e descrição de hardware |
| SO de desenvolvimento (referência TCC) | Ubuntu 24.04.1 LTS | notebook i7-1165G7, 16GB RAM |

---

## 3. Estrutura relevante do repositório

```
robcmp/
├── wrappers/
│   └── tflm/
│       ├── tflm_wrapper.cpp      # camada de interoperabilidade RL <-> TFLM (extern "C")
│       └── kernel_config.h       # flags ENABLE_<KERNEL> e ENABLE_DEBUG_LOGS
├── lib/
│   └── ai/
│       └── tflm.rob              # biblioteca padrão RL: declarações das funções do wrapper
├── platformio/
│   └── platform/
│       └── builder/
│           └── main.py           # orquestrador do build: Robcmp -> link -> firmware -> upload
├── <arquitetura>.ld              # linker scripts existentes: avr5.ld, atmega328p.ld, stm32f1.ld ...
├── <arquitetura>.rob             # descrição de hardware/periféricos: stm32f1.rob (interface intfs.mcu) ...
├── (frontend do compilador)      # .l (Flex), .y/.yy (Bison), classes de AST e análise semântica em C++
└── test/
    └── tflm-tests/               # Makefile de testes que compila o wrapper e roda os modelos de validação
```

> Estrutura acima corresponde ao estado da branch `tcc/tflm-front-end`
> (base `llvm20`). A da branch de trabalho está logo abaixo.

**Estrutura atual (`mestrado/tflm-native`):**

```
src/tinyml/                   # TODA a extensão TinyML do compilador
├── TfliteModel.{h,cpp}       #   análise estática do .tflite (FlatBuffers): kernels,
│                             #   tensores, quantização, o que o TFLM não roda
├── AiModel.{h,cpp}           #   m = ai_model("x.tflite", arena): declaração e geração
├── ModelMember.{h,cpp}       #   resolve m.input0, m.output1.raw, m.input_count ...
├── ModelUse.{h,cpp}          #   ganchos no PropagateTypes que trocam os nós do parser
├── TensorRead / TensorWrite / ModelInvoke   # leitura, escrita, invoke()
├── WrapperCall.{h,cpp}       #   declara e chama as funções do wrapper no IR
└── IntegerConstant.{h,cpp}   #   constantes inteiras e tamanhos estáticos
wrappers/tflm/                # o wrapper C (README.md descreve a API atual)
lib/ai/tflm.rob               # modo biblioteca: declarações do wrapper em RL
lib/stm32f1.{ld,rob}, lib/arch/stm32/   # alvo ARM existente (Cortex-M3)
platformio/platform/builder/main.py      # build pelo PlatformIO
third-party/tflite-micro/     # submódulo do TFLM (gen/ guarda a libtensorflow-microlite.a)
test/tflm/
├── native-syntax.rob         # ESPECIFICAÇÃO executável da sintaxe nativa (compila inteira)
├── models/                   # 8 modelos de validação; externos/ tem 4 públicos maiores
├── data/                     # imagens e áudios de teste (.bin) e a origem deles
├── validate_wrapper.cpp      # camada 1: wrapper x TFLM puro, byte a byte
├── rob/tcc_*.rob             # camada 2: testes do TCC no modo biblioteca
├── rob/nativo_*.rob          # camada 3: os mesmos e mais, na sintaxe nativa
├── check_abi.py              # assinaturas RL x C em x86_64 e Cortex-M
├── check_tflite.py           # análise do .tflite (com tflite_info.cpp)
├── check_ai_model.py         # erros e programas aceitos da sintaxe nativa (101 casos)
├── rob_embed.py              # embute arquivos binários em .rob
└── demo/                     # demonstração para orientação (reuniao.html, make ...)
test/achados/                 # reproduções dos achados do RESULTADOS.md
```

O que cada teste prova está no cabeçalho do próprio arquivo e no
`test/tflm/Makefile`.

**Arquivos que ainda precisam ser criados nesta pesquisa:**
- `stm32f407.ld` — linker script do STM32F407VET6 (mapa de memória flash/SRAM).
- `stm32f407.rob` — descrição dos registradores/periféricos do STM32F407VET6, implementando `intfs.mcu` (mesmo padrão de `lib/stm32f1.rob`), incluindo o registrador do DWT para contagem de ciclos.
- Build do TFLM e do wrapper para Cortex-M4, e a extensão do `main.py`.

---

## 4. O que já foi desenvolvido no TCC (baseline a preservar/evoluir)

O TCC entregou uma **prova de conceito funcional, mas com arquitetura a ser
refeita**. Pontos-chave que o Claude Code deve conhecer antes de mexer no código:

### 4.1 Wrapper C (`tflm_wrapper.cpp`)
- Arquivo `.cpp`, mas todas as funções expostas usam `extern "C"` (C linkage) para permitir ligação em tempo de link com o objeto gerado pelo Robcmp.
- Struct `TFLM_Instance` encapsula `interpreter`, `allocator` e `resolver` do TFLM.
- Enum `KernelType` (espelha `BuiltinOperator` do TFLM) + função `RegisterOp(resolver, kernel_type)` — implementada como **um único `switch-case` monolítico**, com inclusão/exclusão de kernels via diretivas `#if ENABLE_<KERNEL>` em `kernel_config.h`. **Esta é a principal fragilidade a ser refatorada** (ver seção 5.1).
- API principal exposta ao `.rob`:
  - `InitializeInterpreter(model_data, tensor_arena, required_kernels, ..., tensor_arena_size, num_kernels) -> uintptr_t` (handle, pois RL não tem ponteiros nativos)
  - `GetInputTensor(handle, index, ...)` / `GetOutputTensor(handle, index, ...)`
  - `SetTensorValue` / `SetTensorArray` / `SetTensorFromIntArray` (conversão automática float ↔ quantizado, fórmula: `q = round(f/scale + zero_point)`, `f = scale*(q - zero_point)`)
  - `GetTensorAsFloat` / `GetTensorArray` (desquantização automática)
  - `InvokeInterpreter(handle, ...)`
  - `DestroyInterpreter(handle, ...)`
  - **Nota:** funções recebem parâmetros extras "sem nome" só por compatibilidade de assinatura com o backend LLVM/convenção de vetores do Robcmp (para cada argumento vetor, a RL injeta automaticamente um argumento inteiro de tamanho). Isso é uma convenção do compilador, não um capricho do wrapper — deve ser respeitada em qualquer nova função exposta.

### 4.2 Biblioteca padrão `ai.tflm` (`lib/ai/tflm.rob`)
- Arquivo em RL que apenas **declara** (não implementa) as funções do wrapper.
- Uso: `use ai.tflm` no início do programa `.rob`.
- Serve de "fachada": a implementação real está na `.a` estática do wrapper; a resolução acontece em tempo de linkedição.
- **Este modo de uso deve continuar funcionando após todas as mudanças** — é o modo "biblioteca padrão", que coexistirá com o novo modo "sintaxe nativa".

### 4.3 Pipeline de build (estado anterior — fragmentado)
Três artefatos compilados separadamente e depois linkados:
1. `programa.o` — gerado pelo Robcmp a partir do `.rob` do usuário.
2. `wrapper.a` — gerado por um Makefile em `test/tflm-tests`, compilando `tflm_wrapper.cpp` com Clang.
3. `libtflm.a` (`libtensorflow-microlite.a`) — gerado pelo sistema de build próprio do TFLM.

### 4.4 Validação usada no TCC (deve ser replicada no mestrado — seção 6.1.4 do projeto)
5 modelos TinyML de referência (todos do repositório oficial do TFLM, exceto o classificador de spam, do Kaggle):
1. **Sine Predictor** ("hello world" do TinyML — regressão simples)
2. **Sine Predictor Quantizado** (pós-treinamento, pesos/ativações int8)
3. **Spam Classifier** (NLP — entrada `int32[20]`, saída `float32[2]`)
4. **Micro Speech** (CNN <20kB, espectrograma de 49 features, 4 classes de saída)
5. **Person Detection** (~250kB, entrada `int8` escala de cinza, saída pessoa/não-pessoa)

Critérios de teste: corretude funcional (unit tests) + comparação de **tempo de execução** (Hyperfine) e **tamanho de binário** entre a abordagem RL/`ai.tflm` e a implementação equivalente em C++ puro com TFLM.

---

## 5. O que precisa ser feito nesta pesquisa (mestrado)

### 5.1 Refatoração do wrapper (`tflm_wrapper.cpp`)
- Decompor `RegisterOp` (atualmente um `switch-case` monolítico) em **funções unitárias por kernel** — ex.: `add_fully_connected(resolver, model_instance)` — cada uma com responsabilidade única.
- Eliminar o enum `KernelType` e as diretivas manuais `#define ENABLE_<KERNEL>`.
- Objetivo: que apenas os kernels **efetivamente referenciados** pelo `.rob` do usuário sejam linkados (via flags de linkedição), sem precisar de configuração manual em `kernel_config.h`.

### 5.2 Modificação do frontend (léxico, sintático, semântico)
- **Léxico (`.l`):** manter a maioria dos tokens da versão anterior; **remover/substituir** os tokens `.input` e `.output` (colidem com nomes de atributos de tipos definidos pelo usuário — bug conhecido a corrigir).
- **Sintático:** adaptar a gramática à branch mais atual do Robcmp (não é a mesma branch usada no TCC). Mudança central: **eliminar o nó único `ModelNode`** (antipadrão *Large Class*, Fowler) que atualmente trata declaração, invocação, entrada, saída e destruição de modelo com um único parâmetro identificador.
- **Semântico:** decompor `ModelNode` em módulos coesos, um por operação:
  - declaração do modelo
  - invocação da inferência
  - passagem de valores aos tensores de entrada
  - resgate de valores dos tensores de saída
  - destruição da instância
- **Novo módulo:** análise estática de `.tflite` via FlatBuffers (`schema_generated.h`), executada em tempo de compilação como fase de elaboração prévia (não percorre a AST nem gera IR diretamente). Extrai: kernels necessários, tipos de tensor de entrada/saída, `scale`/`zero_point`. Objetivo: mover verificações que hoje falham só em runtime (kernel ausente, tipo incompatível) para **erros de compilação**.

### 5.3 Pipeline de build unificado
- Ponto de entrada preservado: `robcmp -a <arquitetura> -o programa.o <fonte.rob>`.
- Estender `platformio/platform/builder/main.py` para:
  1. Reconhecer `STM32F407VET6` como alvo válido, associando automaticamente `stm32f407.ld` e `stm32f407.rob`.
  2. Incluir automaticamente `wrapper.a` e `libtensorflow-microlite.a` na linkedição quando o programa usar `ai.tflm` ou a nova sintaxe nativa de TinyML.
  3. Derivar transparentemente os parâmetros de compilação cruzada (`-a` do Robcmp, flags de linker ARM Cortex-M) a partir do `platformio.ini` do projeto do usuário.
- `wrapper.a` e `libtensorflow-microlite.a` devem ser **disponibilizados pré-compilados** no repositório (x86_64 e ARM Cortex-M4), para que o desenvolvedor final não precise compilar o TFLM manualmente.

### 5.4 Validação e experimentação (critérios de aceite)
- Reexecutar os 5 modelos do TCC + novos modelos adicionais, cobrindo:
  - modo `ai.tflm` (biblioteca padrão) — testar retrocompatibilidade.
  - modo sintaxe nativa (novo).
  - ambiente nativo (x86_64) e embarcado (STM32F407VET6).
- Um modelo é considerado validado quando os testes unitários passam **nos dois ambientes**.
- Métricas a coletar por modelo/modalidade/plataforma: **tamanho do binário** e **tempo de execução** (Hyperfine em x86_64; contador de ciclos DWT no STM32).

---

## 6. Convenções de engenharia a seguir

- **Responsabilidade única / Clean Architecture** (Martin, 2008) — guia explícito da refatoração do wrapper e do frontend. Evitar classes/funções "God object" como o antigo `ModelNode` ou o antigo `RegisterOp`.
- **Extensão não-intrusiva da gramática** — não alterar semântica ou sintaxe já estabelecida da RL fora do escopo de TinyML.
- **Deslocar verificações de runtime para compile-time sempre que possível** (mesmo princípio já adotado no mecanismo de injeção de dependência dirigida pelo compilador do Robcmp — Oliveira & Costa, 2025).
- Qualquer nova função exposta pelo wrapper para uso via `.rob` deve respeitar a convenção de assinatura do Robcmp (parâmetro de tamanho extra para cada vetor). Handles, índices e contagens usam `uint64_t` no C — nunca `uintptr_t` nem `size_t`, que têm 32 bits no Cortex-M enquanto o `uint64` da RL tem 64. O `make -C test/tflm check` verifica isso para todas as funções. Nenhum parâmetro escalar `int8_t`/`int16_t`: o robcmp não marca `signext` e não estende o sinal, e uma função C compilada pelo clang lê −1 como 255 (`make -s -C test/tflm/demo sinal`). Use `int32_t` e converta dentro do wrapper, como em `tflm_set_raw_value`. Retornos `int8_t` são seguros.
- **Defeitos da RL base não são corrigidos nesta pesquisa.** Encontrado um defeito fora de `src/tinyml` (falta de `signext`, `or` entre comparações constantes, literal `-128` inferido como `int16`...), registre-o no `RESULTADOS.md` e contorne-o no TinyML ou nos testes. Mudança no núcleo do compilador só com decisão explícita do autor.
- Preservar a possibilidade de uso **tanto** via biblioteca padrão `ai.tflm` **quanto** via sintaxe nativa — não são mutuamente exclusivos.
- **A sintaxe vem antes da implementação.** Toda construção nova da linguagem é definida primeiro em `test/tflm/native-syntax.rob` — a especificação executável da sintaxe nativa — e só depois implementada no wrapper e no compilador. O que o compilador emite decorre da sintaxe, e o formato das funções do wrapper decorre do que o compilador emite; implementar antes de especificar inverte essa dependência.

---

## 7. Comandos úteis

Verificados na branch `mestrado/tflm-native`:

```bash
# Compilar o robcmp (CMake em build/; regera o parser se o .y/.l mudar)
make -C build -j$(nproc)

# Validação completa do TinyML: wrapper, modo biblioteca, sintaxe nativa,
# assinaturas, análise do .tflite e erros de compilação. Na primeira vez
# compila a libtensorflow-microlite.a (minutos).
make -C test/tflm check
make -C test/tflm nativo      # só os testes da sintaxe nativa
make -C test/tflm ai_model    # só os casos de compilação (check_ai_model.py)

# Inspecionar um modelo: tensores, quantização, kernels, o que o TFLM não roda
test/tflm/tflite_info modelo.tflite

# Compilar um programa .rob (sem -o, imprime o LLVM IR otimizado)
build/robcmp -I lib -o programa.o fonte.rob      # host x86_64
build/robcmp -a stm32f1 -o programa.o fonte.rob  # Cortex-M3, o alvo ARM que existe
build/robcmp -Od fonte.rob                       # IR antes de otimizar
# ligar com o wrapper e o TFLM (host):
g++ programa.o test/tflm/out/libtflm_wrapper.a \
    third-party/tflite-micro/gen/linux_x86_64_default_gcc/lib/libtensorflow-microlite.a \
    -Wl,--gc-sections -o programa

# Demonstração (ver test/tflm/demo/reuniao.html)
make -s -C test/tflm/demo exemplo   # também: ir, erros, testes, memoria, sinal

# Compilador ARM instalado pelo PlatformIO (para o Cortex-M4)
ls ~/.platformio/packages/toolchain-gccarmnoneeabi*/bin/arm-none-eabi-g++

# Build/upload via PlatformIO (main.py como orquestrador) -- ainda não para o F407
pio run
pio run -t upload

# Benchmark nativo (x86_64)
hyperfine --warmup 10 -m 100 './firmware'
```

> **Sessões na nuvem (Claude Code):** antes de compilar ou testar, rode
> `robcmp-preparar` na raiz do repositório. O comando é instalado pelo setup
> script do ambiente: inicializa o submódulo do TFLM, copia o TFLM
> pré-compilado de `/opt/tflm` e compila `build/robcmp` (~1min30s).

---

## 8. Fora de escopo deste arquivo

Referencial teórico completo, revisão de trabalhos relacionados (TFLM,
STM32Cube.AI, AIfES, MicroFlow), justificativa metodológica formal,
cronograma e referências bibliográficas — todo esse conteúdo pertence ao
projeto de pesquisa/dissertação, não ao contexto de codificação. Consulte os
documentos originais (TCC e projeto de mestrado) para esse material.
