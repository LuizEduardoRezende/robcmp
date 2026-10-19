# Wrapper TFLM

Camada C entre a Robotics Language (RL) e o TensorFlow Lite Micro (TFLM). As
funções têm ligação C (`extern "C"`) e são chamadas de dois jeitos:

- **modo biblioteca:** o programa RL faz `use ai.tflm;` e chama as funções
  pelo nome; as declarações estão em `lib/ai/tflm.rob`;
- **sintaxe nativa:** o programa declara `m = ai_model("modelo.tflite", arena)`
  e o compilador emite as chamadas (`src/tinyml/WrapperCall.cpp`), com as
  mesmas assinaturas do modo biblioteca.

| Arquivo | Conteúdo |
|---|---|
| `tflm_wrapper.h` | a API, com as convenções de assinatura |
| `tflm_wrapper.cpp` | a implementação |

A antiga configuração de kernels por `kernel_config.h` (TCC) não existe mais:
cada kernel é uma função própria, e o linker só traz os que o programa chama.

## A API

| Grupo | Funções | Na sintaxe nativa |
|---|---|---|
| Ciclo de vida | `tflm_create_resolver`, `tflm_create_model`, `tflm_free_model` | `m = ai_model(...)`; não há destruição |
| Kernels | `tflm_add_<kernel>`, um por kernel: 119, entre builtins e custom ops | emitidos pelo compilador, um por kernel do `.tflite` |
| Execução | `tflm_invoke` | `m.invoke()` |
| Tensores | `tflm_input`, `tflm_output`, `tflm_input_count`, `tflm_output_count`, `tflm_tensor_size` | `m.input0`, `m.output1`; contagens e tamanhos são constantes de compilação |
| Valores reais | `tflm_set_value(s)`, `tflm_get_value(s)`, com conversão float ↔ quantizado | `m.input0 = v`, `x = m.output0[k]` |
| Códigos crus | `tflm_set_raw_int8/int16`, `tflm_get_raw_int8/int16`, `tflm_set_raw_value`, `tflm_get_raw_value` | `m.input0.raw = c`, `c = m.output0.raw[k]` |
| Utilitário | `tflm_print_float` | — |

O ciclo de vida tem três fases: criar o resolver, registrar cada kernel que o
modelo usa e criar o modelo sobre o resolver. É a fase do meio que permite ao
linker descartar os kernels não usados (`-ffunction-sections` com
`--gc-sections`).

## Convenções

- **Handles, índices e contagens são `uint64_t`**, o `uint64` da RL em
  qualquer alvo. Nunca `uintptr_t` nem `size_t`, que têm 32 bits no Cortex-M.
- **Para cada vetor, a RL injeta um `int32_t` com o número de elementos**,
  depois de todos os parâmetros, na ordem dos vetores. Esses parâmetros não
  aparecem na declaração `.rob`.
- **Nenhum parâmetro escalar `int8_t`/`int16_t`.** O robcmp não marca
  `signext`, e uma função compilada pelo clang lê −1 como 255
  (`make -s -C test/tflm/demo sinal`). Use `int32_t`, como `tflm_set_raw_value`.
- **Retorno `int8_t`:** 0 é sucesso, −1 é falha.
- **Sem heap.** Os intérpretes vivem numa tabela estática e o TFLM aloca
  dentro da arena do modelo.

`make -C test/tflm abi` confere, em x86_64 e Cortex-M, que cada função tem a
mesma assinatura em `lib/ai/tflm.rob` e em `tflm_wrapper.h`.

## Parâmetros de compilação

| Define | Padrão | Efeito |
|---|---|---|
| `TF_LITE_STATIC_MEMORY` | **obrigatório** | o TFLM é compilado com ele, e o define muda o layout de `TfLiteTensor`; sem ele, o wrapper lê campos em posições erradas, sem erro de compilação |
| `TFLM_MAX_MODELS` | 4 | instâncias de intérprete na tabela estática. O compilador tem o mesmo número (`src/tinyml/AiModel.cpp`); mudar um exige mudar o outro, e o `check_ai_model.py` falha se divergirem |
| `TFLM_MAX_OPS` | 128 | kernels que cada modelo pode registrar (há 119). É o tamanho do resolver de cada instância: ~4,8 KB de RAM por instância no Cortex-M4, ~9,1 KB no x86_64 |
| `TFLM_DEBUG_LOGS` | 0 | 1 liga os logs de diagnóstico (`TFLM_LOG`) |

## Compilar e testar

```bash
make -C test/tflm out/libtflm_wrapper.a   # o wrapper, para x86_64
make -C test/tflm check                   # toda a validação
```

O `test/tflm/Makefile` compila o wrapper com `g++`, o mesmo compilador da
`libtensorflow-microlite.a`, e as flags certas. A camada 1 da validação
(`test/tflm/validate_wrapper.cpp`) roda o wrapper sem a RL e compara cada saída
byte a byte com o TFLM usado diretamente.

Ainda não há build para Cortex-M4: é o passo seguinte da pesquisa.
