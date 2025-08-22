#include "wrappers/tflm/tflm_wrapper.h"             // Inclui o cabeçalho do seu wrapper
#include "wrappers/tflm/hello_world_model_data.h"   // Inclui o modelo convertido pelo xxd
#include <stdio.h>

// xxd -i third_party/tflite-micro/tensorflow/lite/micro/examples/hello_world/models/hello_world_int8.tflite > hello_world_model_data.h

// 1. Aloca a memória para a arena de forma estática, como a aplicação final faria
constexpr int ARENA_SIZE = 10 * 1024; // 10KB (ajuste conforme necessário)
alignas(16) static uint8_t tensor_arena[ARENA_SIZE];

int main() {
    printf("--- Iniciando Teste do Wrapper TFLM ---\n");

    // O xxd cria uma variável com o nome do arquivo, substituindo pontos por underscores
    // e adicionando "_len" para o tamanho.
    const unsigned char* model_data = g_hello_world_model_data; // Dados do modelo
    const unsigned int model_size = g_hello_world_model_data_len;

    // 2. Chama a função de inicialização do wrapper
    void* model_handle = InitializeInterpreter(model_data, tensor_arena, ARENA_SIZE);

    if (model_handle == nullptr) {
        printf("TESTE FALHOU: A inicialização retornou NULL.\n");
        return -1;
    }
    printf("Teste de Inicialização: SUCESSO.\n");

    // 3. Obtém o tensor de entrada (apenas para verificar se não é nulo)
    TfLiteTensor* input_tensor = GetInputTensor(model_handle, 0);
    if (input_tensor == nullptr) {
        printf("TESTE FALHOU: GetInputTensor retornou NULL.\n");
        return -1;
    }
    printf("Teste GetInputTensor: SUCESSO. Tipo do tensor: %d\n", input_tensor->type);
    input_tensor->data.f[0] = 1.23;

    // 4. Invoca o interpretador
    printf("Invocando o interpretador...\n");
    InvokeInterpreter(model_handle);
    printf("Invocação concluída.\n");

    // 5. Obtém o tensor de saída (apenas para verificar)
    const TfLiteTensor* output_tensor = GetOutputTensor(model_handle, 0);
     if (output_tensor == nullptr) {
        printf("TESTE FALHOU: GetOutputTensor retornou NULL.\n");
        return -1;
    }
    printf("Teste GetOutputTensor: SUCESSO.\n");
    printf("Resultado: %f\n", output_tensor->data.f[0]);

    // 6. Destrói a instância
    DestroyInterpreter(model_handle);
    printf("--- Teste do Wrapper TFLM Concluído ---\n");

    return 0;
}