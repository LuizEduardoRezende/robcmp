#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/firmware_spam_detector/models/spam_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
using SpamDetectorOpResolver = tflite::MicroMutableOpResolver<4>;

TfLiteStatus RegisterOps(SpamDetectorOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddGather());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  return kTfLiteOk;
}
}  // namespace

// Global variables for tests
constexpr int kTensorArenaSize = 4 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
SpamDetectorOpResolver op_resolver;

// Test sentences
int32_t test_sentence_1[20] = {1, 82, 70, 214, 55, 98, 84, 47, 89, 26, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // "This is a great article thank you for the information"
int32_t test_sentence_2[20] = {1, 93, 326, 72, 8, 485, 24, 35, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // "win free money online click www site com"

// Test initialization
int test_init() {
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));
  
  const tflite::Model* model = ::tflite::GetModel(g_spam_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return 1;
  }
  
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    return 1;
  }
  
  return 0;
}

// Test inference for non-spam sentence
int test_infer_non_spam() {
  const tflite::Model* model = ::tflite::GetModel(g_spam_model_data);
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
  
  // Set non-spam sentence
  for (int j = 0; j < 20; ++j) {
    input->data.i32[j] = test_sentence_1[j];
  }
  
  if (interpreter.Invoke() != kTfLiteOk) {
    return 2;
  }
  
  float not_spam_prob = output->data.f[0];
  float spam_prob = output->data.f[1];
  
  MicroPrintf("test_infer_non_spam: not_spam_prob=%f, spam_prob=%f", (double)not_spam_prob, (double)spam_prob);
  
  // Expect non-spam probability to be higher
  if (spam_prob > not_spam_prob) {
    return 3;
  }
  
  return 0;
}

// Test inference for spam sentence
int test_infer_spam() {
  const tflite::Model* model = ::tflite::GetModel(g_spam_model_data);
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
  
  // Set spam sentence
  for (int j = 0; j < 20; ++j) {
    input->data.i32[j] = test_sentence_2[j];
  }
  
  if (interpreter.Invoke() != kTfLiteOk) {
    return 2;
  }
  
  float not_spam_prob = output->data.f[0];
  float spam_prob = output->data.f[1];
  
  MicroPrintf("test_infer_spam: not_spam_prob=%f, spam_prob=%f", (double)not_spam_prob, (double)spam_prob);
  
  // Expect spam probability to be higher
  if (not_spam_prob > spam_prob) {
    return 3;
  }
  
  return 0;
}

// Test failure when getting input tensor (invalid index)
int test_input_tensor_fail() {
  const tflite::Model* model = ::tflite::GetModel(g_spam_model_data);
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
  const tflite::Model* model = ::tflite::GetModel(g_spam_model_data);
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
  if (test_infer_non_spam() != 0) { result = 2; }
  if (test_infer_spam() != 0) { result = 3; }
  if (test_input_tensor_fail() != 0) { result = 4; }
  if (test_output_tensor_fail() != 0) { result = 5; }
  
  MicroPrintf("Test result: %d", result);
  if (result == 0) {
    MicroPrintf("All tests passed successfully!");
  }
  
  return result;
}