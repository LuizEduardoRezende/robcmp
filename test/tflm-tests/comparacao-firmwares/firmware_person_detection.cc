#include <cstdint>
#include <cstring>
#include <cstdio>

// Includes do TFLM
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Includes do Modelo e Dados
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "tensorflow/lite/micro/examples/person_detection/testdata/no_person_image_data.h"
#include "tensorflow/lite/micro/examples/person_detection/testdata/person_image_data.h"
#include "tensorflow/lite/micro/models/person_detect_model_data.h"

// Definição da Arena (Mesmo tamanho: 90KB)
constexpr int kTensorArenaSize = 90000;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

int main(int argc, char* argv[]) {
  // 1. Inicialização do Modelo
  const tflite::Model* model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("Erro: Schema version mismatch.\n");
    return 1;
  }

  // 2. Registro de Operações
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  // 3. Instanciação do Interpretador
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // 4. Alocação de Memória
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    printf("Erro: AllocateTensors failed.\n");
    return 1;
  }

  // Obtém ponteiros para entrada e saída
  TfLiteTensor* input = interpreter->input(0);
  TfLiteTensor* output = interpreter->output(0);

  memcpy(input->data.int8, g_person_image_data, input->bytes);

  // Executa Inferência
  if (interpreter->Invoke() != kTfLiteOk) {
    printf("Erro: Invoke failed (Person).\n");
    return 1;
  }

  // Verifica Resultado
  int8_t person_score = output->data.int8[kPersonIndex];
  int8_t no_person_score = output->data.int8[kNotAPersonIndex];

  if (person_score > no_person_score) {
    printf("Passou: Pessoa detectada.\n");
  } else {
    printf("Falha: Pessoa não detectada.\n");
    return 1;
  }

  memcpy(input->data.int8, g_no_person_image_data, input->bytes);

  if (interpreter->Invoke() != kTfLiteOk) {
    printf("Erro: Invoke failed (No Person).\n");
    return 1;
  }

  person_score = output->data.int8[kPersonIndex];
  no_person_score = output->data.int8[kNotAPersonIndex];

  if (no_person_score > person_score) {
    printf("Passou: Não-Pessoa detectada.\n");
  } else {
    printf("Falha: Falso positivo (Girafa detectada como pessoa).\n");
    return 1;
  }

  return 0; 
}