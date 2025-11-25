#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/firmware_hello_world_float/models/hello_world_float_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
using HelloWorldOpResolver = tflite::MicroMutableOpResolver<1>;

TfLiteStatus RegisterOps(HelloWorldOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  return kTfLiteOk;
}
}  // namespace

// Global variables for tests
constexpr int kTensorArenaSize = 3000;
uint8_t tensor_arena[kTensorArenaSize];
HelloWorldOpResolver op_resolver;

// Test initialization
int test_init() {
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));
  
  const tflite::Model* model = ::tflite::GetModel(g_hello_world_float_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return 1;
  }
  
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    return 1;
  }
  
  return 0;
}

// Test inference for input 0.0
int test_infer_zero() {
  const tflite::Model* model = ::tflite::GetModel(g_hello_world_float_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return 1;
  }
  
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    return 1;
  }
  
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  
  if (input == nullptr || output == nullptr) {
    return 2;
  }
  
  float input_value = 0.0f;
  input->data.f[0] = input_value;
  
  if (interpreter.Invoke() != kTfLiteOk) {
    return 2;
  }
  
  float result = output->data.f[0];
  MicroPrintf("test_infer_zero: x=0.0, y=%f", (double)result);
  
  // Verifica se o resultado está próximo de 0 (seno de 0)
  if (result < -0.5f || result > 0.5f) {
    return 3;
  }
  
  return 0;
}

// Test inference for input 1.57 (π/2)
int test_infer_pi_2() {
  const tflite::Model* model = ::tflite::GetModel(g_hello_world_float_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return 1;
  }
  
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    return 1;
  }
  
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  
  if (input == nullptr || output == nullptr) {
    return 2;
  }
  
  float input_value = 1.57f;
  input->data.f[0] = input_value;
  
  if (interpreter.Invoke() != kTfLiteOk) {
    return 2;
  }
  
  float result = output->data.f[0];
  MicroPrintf("test_infer_pi_2: x=1.57, y=%f", (double)result);
  
  // Verifica se o resultado está próximo de 1 (seno de PI/2)
  if (result < 0.5f || result > 1.5f) {
    return 3;
  }
  
  return 0;
}

// Test failure when getting input tensor (invalid index)
int test_input_tensor_fail() {
  const tflite::Model* model = ::tflite::GetModel(g_hello_world_float_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return 1;
  }
  
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    return 1;
  }
  
  TfLiteTensor* input = interpreter.input(99); // invalid index
  
  if (input != nullptr) {
    return 2;
  }
  
  return 0;
}

// Test failure when getting output tensor (invalid index)
int test_output_tensor_fail() {
  const tflite::Model* model = ::tflite::GetModel(g_hello_world_float_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return 1;
  }
  
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    return 1;
  }
  
  TfLiteTensor* output = interpreter.output(99); // invalid index
  
  if (output != nullptr) {
    return 2;
  }
  
  return 0;
}

// Main function that runs all tests
int main(int argc, char* argv[]) {
  tflite::InitializeTarget();
  
  int result = 0;
  
  if (test_init() != 0) { result = 1; }
  if (test_infer_zero() != 0) { result = 2; }
  if (test_infer_pi_2() != 0) { result = 3; }
  if (test_input_tensor_fail() != 0) { result = 4; }
  if (test_output_tensor_fail() != 0) { result = 5; }
  
  MicroPrintf("Test result: %d", result);
  if (result == 0) {
    MicroPrintf("All tests passed successfully!!!");
  }
  
  return result;
}