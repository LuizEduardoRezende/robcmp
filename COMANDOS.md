# Comandos do robcmp

Referência rápida para compilar o compilador, usá-lo, rodar os testes e conferir
as versões das dependências. Todos os comandos rodam a partir da raiz do
repositório.

---

## 1. Compilar o compilador

### Primeira vez numa máquina nova

Baixar o submódulo do TFLM:

```bash
git submodule update --init third-party/tflite-micro
```

Baixar as dependências do TFLM — inclui o FlatBuffers 25.9, que o robcmp exige
para o módulo de análise de `.tflite` (`src/tinyml/`):

```bash
make -C third-party/tflite-micro -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads
```

Configurar o build. Refazer sempre que um `.cpp` novo for criado em `src/` (o
CMake só descobre arquivos novos ao reconfigurar):

```bash
cmake -S . -B build
```

### No dia a dia

```bash
cmake --build build -j$(nproc)
```

O executável fica em `build/robcmp`. O léxico (`src/Language.l`) e o parser
(`src/Language.y`) são regenerados pelo Flex e pelo Bison automaticamente.

### Biblioteca do TFLM

A `libtensorflow-microlite.a` é compilada sozinha pelo `make -C test/tflm check`
na primeira vez. Para compilar manualmente:

```bash
make -C third-party/tflite-micro -f tensorflow/lite/micro/tools/make/Makefile microlite -j$(nproc)
```

---

## 2. Usar o robcmp

Sem argumentos, imprime a sintaxe e a lista de alvos:

```bash
./build/robcmp
```

| Opção | O que faz |
|---|---|
| `-a <alvo>` | Arquitetura de destino. Sem `-a`, compila para o próprio PC (`native`). |
| `-I <pasta>` | Onde procurar os arquivos de `use`. Normalmente `-I lib`. |
| `-o <saida.o>` | Gera o objeto. **Sem `-o`, imprime o LLVM IR na tela.** |
| `-O0 -O1 -O2 -O3 -Os -Oz` | Otimização. Padrão: `-Oz` (menor tamanho). |
| `-g` | Informação de depuração. |
| `-bdep` | Compila também o corpo dos arquivos de `use`, pela gramática completa. Sem ela, o `use` lê só as declarações, por uma gramática simplificada (`src/LanguageUse.y`). |
| `-s <arquivo.spec>` | Arquivo de *bind* da injeção de dependência (ex.: `bind b3 to display.reset singleton;`). |

Alvos aceitos: `native`, `attiny13a`, `attiny85`, `atmega328p`, `atmega32u4`,
`atmega1284p`, `stm32f1`, `esp32`. O STM32F407 ainda não existe (Etapa 4).

### Compilar e rodar no PC

```bash
./build/robcmp -I lib -o programa.o programa.rob
```

```bash
cc -Wl,--gc-sections -no-pie programa.o -o programa
```

### Ver o LLVM IR gerado

```bash
./build/robcmp -I lib programa.rob > programa.ll
```

### Compilar para um microcontrolador

```bash
./build/robcmp -a stm32f1 -I lib -o programa.o programa.rob
```

### Programa com TinyML no modo biblioteca (`use ai.tflm`)

O objeto é linkado com o wrapper e com o TFLM. O `make -C test/tflm check` gera
as duas bibliotecas.

```bash
g++ programa.o test/tflm/out/libtflm_wrapper.a third-party/tflite-micro/gen/linux_x86_64_default_gcc/lib/libtensorflow-microlite.a -Wl,--gc-sections -o programa
```

---

## 3. Testes

Testes gerais do compilador (65 programas; 63 passam — `bool.rob` e
`scalar-in-two-contexts.rob` já falhavam antes do mestrado):

```bash
make -C test/general
```

Validação completa de TinyML — wrapper comparado ao TFLM puro, testes do TCC em
RL, assinaturas RL × C em x86_64 e Cortex-M, e o módulo de análise de `.tflite`:

```bash
make -C test/tflm check
```

Só o módulo de análise de `.tflite`:

```bash
make -C test/tflm tflite
```

O que o compilador extrai de todos os modelos de `test/tflm/models/`:

```bash
make -C test/tflm analyze
```

De um modelo só:

```bash
cd test/tflm && ./tflite_info models/hello_world_int8.tflite
```

---

## 4. Versões das dependências

| Dependência | Versão de referência | Comando |
|---|---|---|
| LLVM | 20.1.8 | `llvm-config --version` |
| Clang | 20.1.8 | `clang++ --version` |
| GCC (robcmp, wrapper e TFLM) | 13.3.0 | `c++ --version` |
| Flex | 2.6.4 | `flex --version` |
| Bison | 3.8.2 | `bison --version` |
| CMake | 3.28.3 | `cmake --version` |
| LLD | 18.1.3 | `ld.lld --version` |
| PlatformIO | 6.1.18 | `pio --version` |
| Hyperfine | 1.18.0 | `hyperfine --version` |
| Python | 3.12.3 | `python3 --version` |
| TFLM (submódulo) | commit `d0318206` | `git submodule status` |
| FlatBuffers | 25.9 | comando abaixo |

Versões medidas em 2026-10-05, Ubuntu 24.04.

O FlatBuffers não tem comando de versão — existe só como headers baixados pelo
TFLM:

```bash
grep -E 'define FLATBUFFERS_VERSION_(MAJOR|MINOR)' third-party/tflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include/flatbuffers/base.h
```

Qual LLVM o build do robcmp está usando de fato:

```bash
grep LLVM_DIR build/CMakeCache.txt
```

**Atenção:** o `ld.lld` instalado é da versão 18, e o LLVM é 20. Hoje não
atrapalha, porque os testes no PC linkam com `cc`/`g++`, mas o `ld.lld` é o
linker usado pelo PlatformIO para o microcontrolador — conferir na Etapa 4.
