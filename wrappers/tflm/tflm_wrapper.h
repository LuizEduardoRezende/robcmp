#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Estrutura opaca para a instância do TensorFlow Lite Micro
typedef struct TFLM_Instance TFLM_Instance;

// Estrutura para representar tensores (forward declaration)
typedef struct TfLiteTensor TfLiteTensor;

// Inicializa um interpretador TensorFlow Lite Micro
// Retorna NULL em caso de erro
void* InitializeInterpreter(const uint8_t* model_data, uint8_t* tensor_arena, size_t tensor_arena_size);

// Destrói e libera a memória de um interpretador
void DestroyInterpreter(void* instance_handle);

// Obtém um tensor de entrada do interpretador
// Retorna NULL em caso de erro
TfLiteTensor* GetInputTensor(void* instance_handle, size_t index);

// Obtém um tensor de saída do interpretador
// Retorna NULL em caso de erro
const TfLiteTensor* GetOutputTensor(void* instance_handle, size_t index);

// Executa uma inferência no interpretador
void InvokeInterpreter(void* instance_handle);

// Executa um benchmark do modelo TensorFlow Lite Micro
// Analisa o modelo, identifica os kernels necessários,
// registra-os automaticamente e executa múltiplas invocações
// Retorna: 0 = sucesso, -1 = schema incompatível, -2 = falha tensores, -3 = falha invocação
int RunBenchmark(const uint8_t* model_data, uint8_t* tensor_arena, size_t tensor_arena_size, int num_invocations);

#ifdef __cplusplus
}
#endif
