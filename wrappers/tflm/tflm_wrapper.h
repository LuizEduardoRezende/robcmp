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

// Enumeração para tipos de kernel suportados
typedef enum {
  ADD = 0,
  AVERAGE_POOL_2D = 1,
  CONCATENATION = 2,
  CONV_2D = 3,
  DEPTHWISE_CONV_2D = 4,
  DEPTH_TO_SPACE = 5,
  DEQUANTIZE = 6,
  EMBEDDING_LOOKUP = 7,
  FLOOR = 8,
  FULLY_CONNECTED = 9,
  HASHTABLE_LOOKUP = 10,
  L2_NORMALIZATION = 11,
  L2_POOL_2D = 12,
  LOCAL_RESPONSE_NORMALIZATION = 13,
  LOGISTIC = 14,
  LSH_PROJECTION = 15,
  LSTM = 16,
  MAX_POOL_2D = 17,
  MUL = 18,
  RELU = 19,
  RELU_N1_TO_1 = 20,
  RELU6 = 21,
  RESHAPE = 22,
  RESIZE_BILINEAR = 23,
  RNN = 24,
  SOFTMAX = 25,
  SPACE_TO_DEPTH = 26,
  SVDF = 27,
  TANH = 28,
  CONCAT_EMBEDDINGS = 29,
  SKIP_GRAM = 30,
  CALL = 31,
  CUSTOM = 32,
  EMBEDDING_LOOKUP_SPARSE = 33,
  PAD = 34,
  UNIDIRECTIONAL_SEQUENCE_RNN = 35,
  GATHER = 36,
  BATCH_TO_SPACE_ND = 37,
  SPACE_TO_BATCH_ND = 38,
  TRANSPOSE = 39,
  MEAN = 40,
  SUB = 41,
  DIV = 42,
  SQUEEZE = 43,
  UNIDIRECTIONAL_SEQUENCE_LSTM = 44,
  STRIDED_SLICE = 45,
  BIDIRECTIONAL_SEQUENCE_RNN = 46,
  EXP = 47,
  TOPK_V2 = 48,
  SPLIT = 49,
  LOG_SOFTMAX = 50,
  DELEGATE = 51,
  BIDIRECTIONAL_SEQUENCE_LSTM = 52,
  CAST = 53,
  PRELU = 54,
  MAXIMUM = 55,
  ARG_MAX = 56,
  MINIMUM = 57,
  LESS = 58,
  NEG = 59,
  PADV2 = 60,
  GREATER = 61,
  GREATER_EQUAL = 62,
  LESS_EQUAL = 63,
  SELECT = 64,
  SLICE = 65,
  SIN = 66,
  TRANSPOSE_CONV = 67,
  SPARSE_TO_DENSE = 68,
  TILE = 69,
  EXPAND_DIMS = 70,
  EQUAL = 71,
  NOT_EQUAL = 72,
  LOG = 73,
  SUM = 74,
  SQRT = 75,
  RSQRT = 76,
  SHAPE = 77,
  POW = 78,
  ARG_MIN = 79,
  FAKE_QUANT = 80,
  REDUCE_PROD = 81,
  REDUCE_MAX = 82,
  PACK = 83,
  LOGICAL_OR = 84,
  ONE_HOT = 85,
  LOGICAL_AND = 86,
  LOGICAL_NOT = 87,
  UNPACK = 88,
  REDUCE_MIN = 89,
  FLOOR_DIV = 90,
  REDUCE_ANY = 91,
  SQUARE = 92,
  ZEROS_LIKE = 93,
  FILL = 94,
  FLOOR_MOD = 95,
  RANGE = 96,
  RESIZE_NEAREST_NEIGHBOR = 97,
  LEAKY_RELU = 98,
  SQUARED_DIFFERENCE = 99,
  MIRROR_PAD = 100,
  ABS = 101,
  SPLIT_V = 102,
  UNIQUE = 103,
  CEIL = 104,
  REVERSE_V2 = 105,
  ADD_N = 106,
  GATHER_ND = 107,
  COS = 108,
  WHERE = 109,
  RANK = 110,
  ELU = 111,
  REVERSE_SEQUENCE = 112,
  MATRIX_DIAG = 113,
  QUANTIZE = 114,
  MATRIX_SET_DIAG = 115,
  ROUND = 116,
  HARD_SWISH = 117,
  IF = 118,
  WHILE = 119,
  NON_MAX_SUPPRESSION_V4 = 120,
  NON_MAX_SUPPRESSION_V5 = 121,
  SCATTER_ND = 122,
  SELECT_V2 = 123,
  DENSIFY = 124,
  SEGMENT_SUM = 125,
  BATCH_MATMUL = 126,
  PLACEHOLDER_FOR_GREATER_OP_CODES = 127,
  CUMSUM = 128,
  CALL_ONCE = 129,
  BROADCAST_TO = 130,
  RFFT2D = 131,
  CONV_3D = 132,
  IMAG = 133,
  REAL = 134,
  COMPLEX_ABS = 135,
  HASHTABLE = 136,
  HASHTABLE_FIND = 137,
  HASHTABLE_IMPORT = 138,
  HASHTABLE_SIZE = 139,
  REDUCE_ALL = 140,
  CONV_3D_TRANSPOSE = 141,
  VAR_HANDLE = 142,
  READ_VARIABLE = 143,
  ASSIGN_VARIABLE = 144,
  BROADCAST_ARGS = 145,
  RANDOM_STANDARD_NORMAL = 146,
  BUCKETIZE = 147,
  RANDOM_UNIFORM = 148,
  MULTINOMIAL = 149,
  GELU = 150,
  DYNAMIC_UPDATE_SLICE = 151,
  RELU_0_TO_1 = 152,
  UNSORTED_SEGMENT_PROD = 153,
  UNSORTED_SEGMENT_MAX = 154,
  UNSORTED_SEGMENT_SUM = 155,
  ATAN2 = 156,
  UNSORTED_SEGMENT_MIN = 157,
  SIGN = 158,
  BITCAST = 159,
  BITWISE_XOR = 160,
  RIGHT_SHIFT = 161,
  STABLEHLO_LOGISTIC = 162,
  STABLEHLO_ADD = 163,
  STABLEHLO_DIVIDE = 164,
  STABLEHLO_MULTIPLY = 165,
  STABLEHLO_MAXIMUM = 166,
  STABLEHLO_RESHAPE = 167,
  STABLEHLO_CLAMP = 168,
  STABLEHLO_CONCATENATE = 169,
  STABLEHLO_BROADCAST_IN_DIM = 170,
  STABLEHLO_CONVOLUTION = 171,
  STABLEHLO_SLICE = 172,
  STABLEHLO_CUSTOM_CALL = 173,
  STABLEHLO_REDUCE = 174,
  STABLEHLO_ABS = 175,
  STABLEHLO_AND = 176,
  STABLEHLO_COSINE = 177,
  STABLEHLO_EXPONENTIAL = 178,
  STABLEHLO_FLOOR = 179,
  STABLEHLO_LOG = 180,
  STABLEHLO_MINIMUM = 181,
  STABLEHLO_NEGATE = 182,
  STABLEHLO_OR = 183,
  STABLEHLO_POWER = 184,
  STABLEHLO_REMAINDER = 185,
  STABLEHLO_RSQRT = 186,
  STABLEHLO_SELECT = 187,
  STABLEHLO_SUBTRACT = 188,
  STABLEHLO_TANH = 189,
  STABLEHLO_SCATTER = 190,
  STABLEHLO_COMPARE = 191,
  STABLEHLO_CONVERT = 192,
  STABLEHLO_DYNAMIC_SLICE = 193,
  STABLEHLO_DYNAMIC_UPDATE_SLICE = 194,
  STABLEHLO_PAD = 195,
  STABLEHLO_IOTA = 196,
  STABLEHLO_DOT_GENERAL = 197,
  STABLEHLO_REDUCE_WINDOW = 198,
  STABLEHLO_SORT = 199,
  STABLEHLO_WHILE = 200,
  STABLEHLO_GATHER = 201,
  STABLEHLO_TRANSPOSE = 202,
  DILATE = 203,
  STABLEHLO_RNG_BIT_GENERATOR = 204,
  REDUCE_WINDOW = 205,
  STABLEHLO_COMPOSITE = 206,
  STABLEHLO_SHIFT_LEFT = 207,
  STABLEHLO_CBRT = 208,
  STABLEHLO_CASE = 209
} KernelType;

// Inicializa um interpretador TensorFlow Lite Micro com kernels específicos
// Retorna 0 em caso de erro, senão retorna handle válido
uintptr_t InitializeInterpreter(const uint8_t* model_data, uint8_t* tensor_arena, const uint8_t* required_kernels, int, int tensor_arena_size, int8_t num_kernels);

// Destrói e libera a memória de um interpretador
void DestroyInterpreter(uintptr_t instance_handle, int);

// Obtém um tensor de entrada do interpretador
// Retorna 0 em caso de erro, senão retorna handle do tensor
uintptr_t GetInputTensor(uintptr_t instance_handle, size_t index, int);

// Obtém um tensor de saída do interpretador
// Retorna 0 em caso de erro, senão retorna handle do tensor
uintptr_t GetOutputTensor(uintptr_t instance_handle, size_t index, int);

// Executa uma inferência no interpretador
void InvokeInterpreter(uintptr_t instance_handle, int);

// Define o valor de um elemento específico em um tensor de entrada
// Converte automaticamente o valor float para o tipo correto do tensor
void SetTensorValue(uintptr_t tensor_handle, size_t index, float value, int);

// Obtém o valor de um elemento específico de um tensor de saída como float
// Converte automaticamente do tipo do tensor para float
float GetTensorAsFloat(uintptr_t tensor_handle, size_t index, int);

// Retorna o número total de elementos em um tensor
size_t GetTensorSize(uintptr_t tensor_handle, int);

// Analisa um modelo TFLite e descobre quais kernels são necessários
// Retorna o número de kernels únicos encontrados
size_t AnalyzeModelKernels(const uint8_t* model_data, uint8_t* required_kernels, size_t max_kernels);

// Executa um benchmark otimizado do modelo TensorFlow Lite Micro
// Usa detecção automática de kernels e executa múltiplas invocações
// Retorna: 0 = sucesso, -1 = erro na inicialização
int RunBenchmarkOptimized(const uint8_t* model_data, int, uint8_t* tensor_arena, int, size_t tensor_arena_size, int num_invocations);

// Função de diagnóstico detalhado do modelo
// Exibe informações completas sobre subgrafos, operadores, tensores, etc.
void DiagnoseModel(const uint8_t* model_data, int);

// Função para verificar integridade do modelo
// Verifica magic number e validações básicas do arquivo .tflite
void VerifyModelData(const uint8_t* model_data, int);

#ifdef __cplusplus
}
#endif
