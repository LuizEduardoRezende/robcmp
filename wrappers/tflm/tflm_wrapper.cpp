#include "third-party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/recording_micro_interpreter.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/recording_micro_allocator.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "third-party/tflite-micro/tensorflow/lite/schema/schema_generated.h"
#include "third-party/tflite-micro/tensorflow/lite/c/common.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/micro_log.h"
#include <cstring>
#include <cstdint>


extern "C" {

using MutableResolver = tflite::MicroMutableOpResolver<100>;

// Enumeração para tipos de kernel possíveis de um modelo
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

void RegisterOp(tflite::MicroMutableOpResolver<100>* resolver, KernelType kernel_type);

struct TFLM_Instance {
    tflite::RecordingMicroInterpreter* interpreter;
    tflite::RecordingMicroAllocator* allocator;
    MutableResolver* resolver;
};

void RegisterOp(tflite::MicroMutableOpResolver<100>* resolver, KernelType kernel_type) {
    switch (kernel_type) {
        case ADD: resolver->AddAdd(); break;
        case AVERAGE_POOL_2D: resolver->AddAveragePool2D(); break;
        case CONCATENATION: resolver->AddConcatenation(); break;
        case CONV_2D: resolver->AddConv2D(); break;
        case DEPTHWISE_CONV_2D: resolver->AddDepthwiseConv2D(); break;
        case DEPTH_TO_SPACE: resolver->AddDepthToSpace(); break;
        case DEQUANTIZE: resolver->AddDequantize(); break;
        case EMBEDDING_LOOKUP: resolver->AddEmbeddingLookup(); break;
        case FLOOR: resolver->AddFloor(); break;
        case FULLY_CONNECTED: resolver->AddFullyConnected(); break;
        case HASHTABLE_LOOKUP: MicroPrintf("AVISO: HASHTABLE_LOOKUP não suportado diretamente"); break;
        case L2_NORMALIZATION: resolver->AddL2Normalization(); break;
        case L2_POOL_2D: resolver->AddL2Pool2D(); break;
        case LOCAL_RESPONSE_NORMALIZATION: MicroPrintf("AVISO: LOCAL_RESPONSE_NORMALIZATION não suportado diretamente"); break;
        case LOGISTIC: resolver->AddLogistic(); break;
        case LSH_PROJECTION: MicroPrintf("AVISO: LSH_PROJECTION não suportado diretamente"); break;
        case LSTM: MicroPrintf("AVISO: LSTM não suportado diretamente"); break;
        case MAX_POOL_2D: resolver->AddMaxPool2D(); break;
        case MUL: resolver->AddMul(); break;
        case RELU: resolver->AddRelu(); break;
        case RELU_N1_TO_1: MicroPrintf("AVISO: RELU_N1_TO_1 não suportado diretamente"); break;
        case RELU6: resolver->AddRelu6(); break;
        case RESHAPE: resolver->AddReshape(); break;
        case RESIZE_BILINEAR: resolver->AddResizeBilinear(); break;
        case RNN: MicroPrintf("AVISO: RNN não suportado diretamente"); break;
        case SOFTMAX: resolver->AddSoftmax(); break;
        case SPACE_TO_DEPTH: resolver->AddSpaceToDepth(); break;
        case SVDF: resolver->AddSvdf(); break;
        case TANH: resolver->AddTanh(); break;
        case CONCAT_EMBEDDINGS: MicroPrintf("AVISO: CONCAT_EMBEDDINGS não suportado diretamente"); break;
        case SKIP_GRAM: MicroPrintf("AVISO: SKIP_GRAM não suportado diretamente"); break;
        case CALL: MicroPrintf("AVISO: CALL não suportado diretamente"); break;
        case CUSTOM: MicroPrintf("AVISO: CUSTOM não suportado diretamente"); break;
        case EMBEDDING_LOOKUP_SPARSE: MicroPrintf("AVISO: EMBEDDING_LOOKUP_SPARSE não suportado diretamente"); break;
        case PAD: resolver->AddPad(); break;
        case UNIDIRECTIONAL_SEQUENCE_RNN: MicroPrintf("AVISO: UNIDIRECTIONAL_SEQUENCE_RNN não suportado diretamente"); break;
        case GATHER: resolver->AddGather(); break;
        case BATCH_TO_SPACE_ND: resolver->AddBatchToSpaceNd(); break;
        case SPACE_TO_BATCH_ND: resolver->AddSpaceToBatchNd(); break;
        case TRANSPOSE: resolver->AddTranspose(); break;
        case MEAN: resolver->AddMean(); break;
        case SUB: resolver->AddSub(); break;
        case DIV: resolver->AddDiv(); break;
        case SQUEEZE: resolver->AddSqueeze(); break;
        case UNIDIRECTIONAL_SEQUENCE_LSTM: resolver->AddUnidirectionalSequenceLSTM(); break;
        case STRIDED_SLICE: resolver->AddStridedSlice(); break;
        case BIDIRECTIONAL_SEQUENCE_RNN: MicroPrintf("AVISO: BIDIRECTIONAL_SEQUENCE_RNN não suportado diretamente"); break;
        case EXP: resolver->AddExp(); break;
        case TOPK_V2: MicroPrintf("AVISO: TOPK_V2 não suportado diretamente"); break;
        case SPLIT: resolver->AddSplit(); break;
        case LOG_SOFTMAX: resolver->AddLogSoftmax(); break;
        case DELEGATE: MicroPrintf("AVISO: DELEGATE não suportado diretamente"); break;
        case BIDIRECTIONAL_SEQUENCE_LSTM: MicroPrintf("AVISO: BIDIRECTIONAL_SEQUENCE_LSTM não suportado diretamente"); break;
        case CAST: resolver->AddCast(); break;
        case PRELU: resolver->AddPrelu(); break;
        case MAXIMUM: resolver->AddMaximum(); break;
        case ARG_MAX: resolver->AddArgMax(); break;
        case MINIMUM: resolver->AddMinimum(); break;
        case LESS: resolver->AddLess(); break;
        case NEG: resolver->AddNeg(); break;
        case PADV2: resolver->AddPadV2(); break;
        case GREATER: resolver->AddGreater(); break;
        case GREATER_EQUAL: resolver->AddGreaterEqual(); break;
        case LESS_EQUAL: resolver->AddLessEqual(); break;
        case SELECT: MicroPrintf("AVISO: SELECT não suportado diretamente"); break;
        case SLICE: resolver->AddSlice(); break;
        case SIN: resolver->AddSin(); break;
        case TRANSPOSE_CONV: resolver->AddTransposeConv(); break;
        case SPARSE_TO_DENSE: MicroPrintf("AVISO: SPARSE_TO_DENSE não suportado diretamente"); break;
        case TILE: MicroPrintf("AVISO: TILE não suportado diretamente"); break;
        case EXPAND_DIMS: resolver->AddExpandDims(); break;
        case EQUAL: resolver->AddEqual(); break;
        case NOT_EQUAL: resolver->AddNotEqual(); break;
        case LOG: resolver->AddLog(); break;
        case SUM: resolver->AddSum(); break;
        case SQRT: resolver->AddSqrt(); break;
        case RSQRT: resolver->AddRsqrt(); break;
        case SHAPE: resolver->AddShape(); break;
        case POW: MicroPrintf("AVISO: POW não suportado diretamente"); break;
        case ARG_MIN: resolver->AddArgMin(); break;
        case FAKE_QUANT: MicroPrintf("AVISO: FAKE_QUANT não suportado diretamente"); break;
        case REDUCE_PROD: MicroPrintf("AVISO: REDUCE_PROD não suportado diretamente"); break;
        case REDUCE_MAX: resolver->AddReduceMax(); break;
        case PACK: resolver->AddPack(); break;
        case LOGICAL_OR: resolver->AddLogicalOr(); break;
        case ONE_HOT: MicroPrintf("AVISO: ONE_HOT não suportado diretamente"); break;
        case LOGICAL_AND: resolver->AddLogicalAnd(); break;
        case LOGICAL_NOT: resolver->AddLogicalNot(); break;
        case UNPACK: resolver->AddUnpack(); break;
        case REDUCE_MIN: resolver->AddReduceMin(); break;
        case FLOOR_DIV: resolver->AddFloorDiv(); break;
        case REDUCE_ANY: MicroPrintf("AVISO: REDUCE_ANY não suportado diretamente"); break;
        case SQUARE: resolver->AddSquare(); break;
        case ZEROS_LIKE: resolver->AddZerosLike(); break;
        case FILL: resolver->AddFill(); break;
        case FLOOR_MOD: resolver->AddFloorMod(); break;
        case RANGE: MicroPrintf("AVISO: RANGE não suportado diretamente"); break;
        case RESIZE_NEAREST_NEIGHBOR: resolver->AddResizeNearestNeighbor(); break;
        case LEAKY_RELU: resolver->AddLeakyRelu(); break;
        case SQUARED_DIFFERENCE: resolver->AddSquaredDifference(); break;
        case MIRROR_PAD: resolver->AddMirrorPad(); break;
        case ABS: resolver->AddAbs(); break;
        case SPLIT_V: resolver->AddSplitV(); break;
        case UNIQUE: MicroPrintf("AVISO: UNIQUE não suportado diretamente"); break;
        case CEIL: resolver->AddCeil(); break;
        case REVERSE_V2: resolver->AddReverseV2(); break;
        case ADD_N: resolver->AddAddN(); break;
        case GATHER_ND: resolver->AddGatherNd(); break;
        case COS: resolver->AddCos(); break;
        case WHERE: MicroPrintf("AVISO: WHERE não suportado diretamente"); break;
        case RANK: MicroPrintf("AVISO: RANK não suportado diretamente"); break;
        case ELU: resolver->AddElu(); break;
        case REVERSE_SEQUENCE: MicroPrintf("AVISO: REVERSE_SEQUENCE não suportado diretamente"); break;
        case MATRIX_DIAG: MicroPrintf("AVISO: MATRIX_DIAG não suportado diretamente"); break;
        case QUANTIZE: resolver->AddQuantize(); break;
        case MATRIX_SET_DIAG: MicroPrintf("AVISO: MATRIX_SET_DIAG não suportado diretamente"); break;
        case ROUND: resolver->AddRound(); break;
        case HARD_SWISH: resolver->AddHardSwish(); break;
        case IF: resolver->AddIf(); break;
        case WHILE: resolver->AddWhile(); break;
        case NON_MAX_SUPPRESSION_V4: MicroPrintf("AVISO: NON_MAX_SUPPRESSION_V4 não suportado diretamente"); break;
        case NON_MAX_SUPPRESSION_V5: MicroPrintf("AVISO: NON_MAX_SUPPRESSION_V5 não suportado diretamente"); break;
        case SCATTER_ND: MicroPrintf("AVISO: SCATTER_ND não suportado diretamente"); break;
        case SELECT_V2: resolver->AddSelectV2(); break;
        case DENSIFY: MicroPrintf("AVISO: DENSIFY não suportado diretamente"); break;
        case SEGMENT_SUM: MicroPrintf("AVISO: SEGMENT_SUM não suportado diretamente"); break;
        case BATCH_MATMUL: resolver->AddBatchMatMul(); break;
        case PLACEHOLDER_FOR_GREATER_OP_CODES: MicroPrintf("AVISO: PLACEHOLDER_FOR_GREATER_OP_CODES não suportado diretamente"); break;
        case CUMSUM: resolver->AddCumSum(); break;
        case CALL_ONCE: resolver->AddCallOnce(); break;
        case BROADCAST_TO: resolver->AddBroadcastTo(); break;
        case RFFT2D: MicroPrintf("AVISO: RFFT2D não suportado diretamente"); break;
        case CONV_3D: MicroPrintf("AVISO: CONV_3D não suportado diretamente"); break;
        case IMAG: MicroPrintf("AVISO: IMAG não suportado diretamente"); break;
        case REAL: MicroPrintf("AVISO: REAL não suportado diretamente"); break;
        case COMPLEX_ABS: MicroPrintf("AVISO: COMPLEX_ABS não suportado diretamente"); break;
        case HASHTABLE: MicroPrintf("AVISO: HASHTABLE não suportado diretamente"); break;
        case HASHTABLE_FIND: MicroPrintf("AVISO: HASHTABLE_FIND não suportado diretamente"); break;
        case HASHTABLE_IMPORT: MicroPrintf("AVISO: HASHTABLE_IMPORT não suportado diretamente"); break;
        case HASHTABLE_SIZE: MicroPrintf("AVISO: HASHTABLE_SIZE não suportado diretamente"); break;
        case REDUCE_ALL: MicroPrintf("AVISO: REDUCE_ALL não suportado diretamente"); break;
        case CONV_3D_TRANSPOSE: MicroPrintf("AVISO: CONV_3D_TRANSPOSE não suportado diretamente"); break;
        case VAR_HANDLE: resolver->AddVarHandle(); break;
        case READ_VARIABLE: resolver->AddReadVariable(); break;
        case ASSIGN_VARIABLE: resolver->AddAssignVariable(); break;
        case BROADCAST_ARGS: resolver->AddBroadcastArgs(); break;
        case RANDOM_STANDARD_NORMAL: MicroPrintf("AVISO: RANDOM_STANDARD_NORMAL não suportado diretamente"); break;
        case BUCKETIZE: MicroPrintf("AVISO: BUCKETIZE não suportado diretamente"); break;
        case RANDOM_UNIFORM: MicroPrintf("AVISO: RANDOM_UNIFORM não suportado diretamente"); break;
        case MULTINOMIAL: MicroPrintf("AVISO: MULTINOMIAL não suportado diretamente"); break;
        case GELU: MicroPrintf("AVISO: GELU não suportado diretamente"); break;
        case DYNAMIC_UPDATE_SLICE: MicroPrintf("AVISO: DYNAMIC_UPDATE_SLICE não suportado diretamente"); break;
        case RELU_0_TO_1: MicroPrintf("AVISO: RELU_0_TO_1 não suportado diretamente"); break;
        case UNSORTED_SEGMENT_PROD: MicroPrintf("AVISO: UNSORTED_SEGMENT_PROD não suportado diretamente"); break;
        case UNSORTED_SEGMENT_MAX: MicroPrintf("AVISO: UNSORTED_SEGMENT_MAX não suportado diretamente"); break;
        case UNSORTED_SEGMENT_SUM: MicroPrintf("AVISO: UNSORTED_SEGMENT_SUM não suportado diretamente"); break;
        case ATAN2: MicroPrintf("AVISO: ATAN2 não suportado diretamente"); break;
        case UNSORTED_SEGMENT_MIN: MicroPrintf("AVISO: UNSORTED_SEGMENT_MIN não suportado diretamente"); break;
        case SIGN: MicroPrintf("AVISO: SIGN não suportado diretamente"); break;
        case BITCAST: MicroPrintf("AVISO: BITCAST não suportado diretamente"); break;
        case BITWISE_XOR: MicroPrintf("AVISO: BITWISE_XOR não suportado diretamente"); break;
        case RIGHT_SHIFT: MicroPrintf("AVISO: RIGHT_SHIFT não suportado diretamente"); break;
        case STABLEHLO_LOGISTIC: MicroPrintf("AVISO: STABLEHLO_LOGISTIC não suportado diretamente"); break;
        case STABLEHLO_ADD: MicroPrintf("AVISO: STABLEHLO_ADD não suportado diretamente"); break;
        case STABLEHLO_DIVIDE: MicroPrintf("AVISO: STABLEHLO_DIVIDE não suportado diretamente"); break;
        case STABLEHLO_MULTIPLY: MicroPrintf("AVISO: STABLEHLO_MULTIPLY não suportado diretamente"); break;
        case STABLEHLO_MAXIMUM: MicroPrintf("AVISO: STABLEHLO_MAXIMUM não suportado diretamente"); break;
        case STABLEHLO_RESHAPE: MicroPrintf("AVISO: STABLEHLO_RESHAPE não suportado diretamente"); break;
        case STABLEHLO_CLAMP: MicroPrintf("AVISO: STABLEHLO_CLAMP não suportado diretamente"); break;
        case STABLEHLO_CONCATENATE: MicroPrintf("AVISO: STABLEHLO_CONCATENATE não suportado diretamente"); break;
        case STABLEHLO_BROADCAST_IN_DIM: MicroPrintf("AVISO: STABLEHLO_BROADCAST_IN_DIM não suportado diretamente"); break;
        case STABLEHLO_CONVOLUTION: MicroPrintf("AVISO: STABLEHLO_CONVOLUTION não suportado diretamente"); break;
        case STABLEHLO_SLICE: MicroPrintf("AVISO: STABLEHLO_SLICE não suportado diretamente"); break;
        case STABLEHLO_CUSTOM_CALL: MicroPrintf("AVISO: STABLEHLO_CUSTOM_CALL não suportado diretamente"); break;
        case STABLEHLO_REDUCE: MicroPrintf("AVISO: STABLEHLO_REDUCE não suportado diretamente"); break;
        case STABLEHLO_ABS: MicroPrintf("AVISO: STABLEHLO_ABS não suportado diretamente"); break;
        case STABLEHLO_AND: MicroPrintf("AVISO: STABLEHLO_AND não suportado diretamente"); break;
        case STABLEHLO_COSINE: MicroPrintf("AVISO: STABLEHLO_COSINE não suportado diretamente"); break;
        case STABLEHLO_EXPONENTIAL: MicroPrintf("AVISO: STABLEHLO_EXPONENTIAL não suportado diretamente"); break;
        case STABLEHLO_FLOOR: MicroPrintf("AVISO: STABLEHLO_FLOOR não suportado diretamente"); break;
        case STABLEHLO_LOG: MicroPrintf("AVISO: STABLEHLO_LOG não suportado diretamente"); break;
        case STABLEHLO_MINIMUM: MicroPrintf("AVISO: STABLEHLO_MINIMUM não suportado diretamente"); break;
        case STABLEHLO_NEGATE: MicroPrintf("AVISO: STABLEHLO_NEGATE não suportado diretamente"); break;
        case STABLEHLO_OR: MicroPrintf("AVISO: STABLEHLO_OR não suportado diretamente"); break;
        case STABLEHLO_POWER: MicroPrintf("AVISO: STABLEHLO_POWER não suportado diretamente"); break;
        case STABLEHLO_REMAINDER: MicroPrintf("AVISO: STABLEHLO_REMAINDER não suportado diretamente"); break;
        case STABLEHLO_RSQRT: MicroPrintf("AVISO: STABLEHLO_RSQRT não suportado diretamente"); break;
        case STABLEHLO_SELECT: MicroPrintf("AVISO: STABLEHLO_SELECT não suportado diretamente"); break;
        case STABLEHLO_SUBTRACT: MicroPrintf("AVISO: STABLEHLO_SUBTRACT não suportado diretamente"); break;
        case STABLEHLO_TANH: MicroPrintf("AVISO: STABLEHLO_TANH não suportado diretamente"); break;
        case STABLEHLO_SCATTER: MicroPrintf("AVISO: STABLEHLO_SCATTER não suportado diretamente"); break;
        case STABLEHLO_COMPARE: MicroPrintf("AVISO: STABLEHLO_COMPARE não suportado diretamente"); break;
        case STABLEHLO_CONVERT: MicroPrintf("AVISO: STABLEHLO_CONVERT não suportado diretamente"); break;
        case STABLEHLO_DYNAMIC_SLICE: MicroPrintf("AVISO: STABLEHLO_DYNAMIC_SLICE não suportado diretamente"); break;
        case STABLEHLO_DYNAMIC_UPDATE_SLICE: MicroPrintf("AVISO: STABLEHLO_DYNAMIC_UPDATE_SLICE não suportado diretamente"); break;
        case STABLEHLO_PAD: MicroPrintf("AVISO: STABLEHLO_PAD não suportado diretamente"); break;
        case STABLEHLO_IOTA: MicroPrintf("AVISO: STABLEHLO_IOTA não suportado diretamente"); break;
        case STABLEHLO_DOT_GENERAL: MicroPrintf("AVISO: STABLEHLO_DOT_GENERAL não suportado diretamente"); break;
        case STABLEHLO_REDUCE_WINDOW: MicroPrintf("AVISO: STABLEHLO_REDUCE_WINDOW não suportado diretamente"); break;
        case STABLEHLO_SORT: MicroPrintf("AVISO: STABLEHLO_SORT não suportado diretamente"); break;
        case STABLEHLO_WHILE: MicroPrintf("AVISO: STABLEHLO_WHILE não suportado diretamente"); break;
        case STABLEHLO_GATHER: MicroPrintf("AVISO: STABLEHLO_GATHER não suportado diretamente"); break;
        case STABLEHLO_TRANSPOSE: MicroPrintf("AVISO: STABLEHLO_TRANSPOSE não suportado diretamente"); break;
        case DILATE: MicroPrintf("AVISO: DILATE não suportado diretamente"); break;
        case STABLEHLO_RNG_BIT_GENERATOR: MicroPrintf("AVISO: STABLEHLO_RNG_BIT_GENERATOR não suportado diretamente"); break;
        case REDUCE_WINDOW: MicroPrintf("AVISO: REDUCE_WINDOW não suportado diretamente"); break;
        case STABLEHLO_COMPOSITE: MicroPrintf("AVISO: STABLEHLO_COMPOSITE não suportado diretamente"); break;
        case STABLEHLO_SHIFT_LEFT: MicroPrintf("AVISO: STABLEHLO_SHIFT_LEFT não suportado diretamente"); break;
        case STABLEHLO_CBRT: MicroPrintf("AVISO: STABLEHLO_CBRT não suportado diretamente"); break;
        case STABLEHLO_CASE: MicroPrintf("AVISO: STABLEHLO_CASE não suportado diretamente"); break;
        default:
            MicroPrintf("AVISO: Tipo de kernel não mapeado: %d", kernel_type);
            break;
    }
}

KernelType MapBuiltinOperatorToKernelType(tflite::BuiltinOperator builtin_op) {
    return (KernelType)builtin_op;
}

uintptr_t InitializeInterpreter(const uint8_t* model_data, uint8_t* tensor_arena, const uint8_t* required_kernels, int, int tensor_arena_size, int8_t num_kernels) {
    MicroPrintf("\n=== InitializeInterpreter ===");
    
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("ERRO: Schema incompatível! Versão: %d, Esperada: %d", 
                   model->version(), TFLITE_SCHEMA_VERSION);
        return 0;
    }

    TFLM_Instance* instance = new TFLM_Instance();
    instance->resolver = new MutableResolver();

    // Registrar kernels convertendo uint8_t para KernelType
      MicroPrintf("Registrando %d kernels necessários.", num_kernels);
    for (int i = 0; i < num_kernels; ++i) {
        KernelType kernel_type = static_cast<KernelType>(required_kernels[i]);
        tflite::BuiltinOperator builtin_op = static_cast<tflite::BuiltinOperator>(kernel_type);
        MicroPrintf("  Kernel[%d]: %d -> %s", i, builtin_op, tflite::EnumNameBuiltinOperator(builtin_op));
        RegisterOp(instance->resolver, kernel_type);
    }  
    // Usar RecordingMicroInterpreter para melhor compatibilidade
    instance->allocator = tflite::RecordingMicroAllocator::Create(tensor_arena, tensor_arena_size);
    if (!instance->allocator) {
        MicroPrintf("ERRO: Falha ao criar RecordingMicroAllocator");
        delete instance->resolver;
        delete instance;
        return 0;
    }
    
    instance->interpreter = new tflite::RecordingMicroInterpreter(
        model, *(instance->resolver), instance->allocator, nullptr);

    MicroPrintf("\n=== VERIFICAÇÃO PRE-ALOCAÇÃO ===");
    MicroPrintf("Arena utilizada antes: %zu bytes", instance->interpreter->arena_used_bytes());
    
    TfLiteStatus allocate_status = instance->interpreter->AllocateTensors();
    MicroPrintf("AllocateTensors status: %d (kTfLiteOk=0)", allocate_status);
    MicroPrintf("Arena utilizada após: %zu bytes", instance->interpreter->arena_used_bytes());
    
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("ERRO: AllocateTensors falhou! Status: %d", allocate_status);
        MicroPrintf("Arena: %d bytes, usada: %zu bytes", tensor_arena_size, instance->interpreter->arena_used_bytes());
        delete instance->interpreter;
        delete instance->resolver;
        delete instance;
        return 0;
    }

    // Debug: Print tensor information (como no exemplo oficial)
    MicroPrintf("\n=== TENSOR DEBUG ===");
    for (size_t i = 0; i < instance->interpreter->inputs_size(); ++i) {
        TfLiteTensor* input_tensor = instance->interpreter->input(i);
        MicroPrintf("Input tensor[%zu]: type=%d, bytes=%zu, data=%p, allocation_type=%d", 
                   i, input_tensor->type, input_tensor->bytes, input_tensor->data.data, input_tensor->allocation_type);
    }
    for (size_t i = 0; i < instance->interpreter->outputs_size(); ++i) {
        TfLiteTensor* output_tensor = instance->interpreter->output(i);
        MicroPrintf("Output tensor[%zu]: type=%d, bytes=%zu, data=%p, allocation_type=%d", 
                   i, output_tensor->type, output_tensor->bytes, output_tensor->data.data, output_tensor->allocation_type);
    }
    MicroPrintf("Arena used: %zu bytes", instance->interpreter->arena_used_bytes());

    // Verificar se os tensores foram alocados corretamente
    bool all_tensors_valid = true;
    
    for (size_t i = 0; i < instance->interpreter->inputs_size(); ++i) {
        TfLiteTensor* tensor = instance->interpreter->input(i);
        if (!tensor || (tensor->bytes > 0 && !tensor->data.data) || tensor->type == kTfLiteNoType) {
            MicroPrintf("ERRO: Input tensor[%zu] inválido - type=%d, bytes=%zu, data=%p", 
                       i, tensor ? tensor->type : -1, tensor ? tensor->bytes : 0, tensor ? tensor->data.data : nullptr);
            all_tensors_valid = false;
        }
    }
    
    for (size_t i = 0; i < instance->interpreter->outputs_size(); ++i) {
        TfLiteTensor* tensor = const_cast<TfLiteTensor*>(instance->interpreter->output(i));
        if (!tensor || (tensor->bytes > 0 && !tensor->data.data) || tensor->type == kTfLiteNoType) {
            MicroPrintf("ERRO: Output tensor[%zu] inválido - type=%d, bytes=%zu, data=%p", 
                       i, tensor ? tensor->type : -1, tensor ? tensor->bytes : 0, tensor ? tensor->data.data : nullptr);
            all_tensors_valid = false;
        }
    }
    
    if (!all_tensors_valid) {
        MicroPrintf("ERRO CRÍTICO: Um ou mais tensores não foram alocados corretamente!");
        MicroPrintf("Isso pode indicar:");
        MicroPrintf("  1. Arena muito pequena (atual: %d bytes, usada: %zu bytes)", tensor_arena_size, instance->interpreter->arena_used_bytes());
        MicroPrintf("  2. Modelo incompatível com esta versão do TensorFlow Lite Micro");
        MicroPrintf("  3. Kernels necessários não registrados");
        MicroPrintf("Sugere-se aumentar o tamanho da arena ou verificar compatibilidade do modelo.");
        delete instance->interpreter;
        delete instance->resolver;
        delete instance;
        return 0;
    }

    MicroPrintf("Interpreter initialized successfully.");
    return reinterpret_cast<uintptr_t>(instance);
}

void DestroyInterpreter(uintptr_t instance_handle, int) {
    if (instance_handle == 0) return;

    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    delete instance->interpreter;
    delete instance->resolver; 
    delete instance;
    MicroPrintf("Interpreter destroyed successfully.");
}

uintptr_t GetInputTensor(uintptr_t instance_handle, size_t index, int) {
    if (instance_handle == 0) {
        MicroPrintf("ERRO: instance_handle nulo");
        return 0;
    }
    
    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    if (instance == nullptr || instance->interpreter == nullptr) {
        MicroPrintf("ERRO: instance ou interpreter nulo");
        return 0;
    }
    
    if (index >= instance->interpreter->inputs_size()) {
        MicroPrintf("ERRO: índice %zu >= %zu", index, instance->interpreter->inputs_size());
        return 0;
    }
    
    TfLiteTensor* tensor = instance->interpreter->input(index);
    if (tensor == nullptr) {
        MicroPrintf("ERRO: tensor nulo");
        return 0;
    }
    
    if (tensor->bytes == 0) {
        MicroPrintf("ERRO: tensor sem dados alocados");
        return 0;
    }
    
    return reinterpret_cast<uintptr_t>(tensor);
}

uintptr_t GetOutputTensor(uintptr_t instance_handle, size_t index, int) {
    if (instance_handle == 0) {
        MicroPrintf("ERRO: instance_handle nulo");
        return 0;
    }
    
    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    if (instance == nullptr || instance->interpreter == nullptr) {
        MicroPrintf("ERRO: instance ou interpreter nulo");
        return 0;
    }
    
    if (index >= instance->interpreter->outputs_size()) {
        MicroPrintf("ERRO: índice %zu fora dos limites", index);
        return 0;
    }
    
    TfLiteTensor* tensor = const_cast<TfLiteTensor*>(instance->interpreter->output(index));
    if (tensor == nullptr) {
        MicroPrintf("ERRO: tensor de saída nulo");
        return 0;
    }
    
    return reinterpret_cast<uintptr_t>(tensor);
}

void SetTensorValue(uintptr_t tensor_handle, size_t index, float value, int) {
    if (tensor_handle == 0) {
        MicroPrintf("ERRO: tensor_handle nulo");
        return;
    }
    
    TfLiteTensor* tensor = reinterpret_cast<TfLiteTensor*>(tensor_handle);
    if (tensor == nullptr) {
        MicroPrintf("ERRO: tensor nulo após cast");
        return;
    }
    
    // Verificar se o tensor está inicializado
    if (tensor->bytes == 0) {
        MicroPrintf("ERRO: tensor não inicializado");
        return;
    }
    
    if (tensor->type == kTfLiteNoType) {
        MicroPrintf("AVISO: Tensor com tipo indefinido - isso pode indicar um problema no modelo");
        MicroPrintf("ERRO: Impossível definir valor em tensor sem tipo definido");
        return;
    }

    // Verificar limites do índice baseado no tipo
    size_t element_count = 0;
    size_t element_size = 0;
    
    switch(tensor->type) {
        case 1: // kTfLiteFloat32
            element_size = sizeof(float);
            break;
        case 9: // kTfLiteInt8  
            element_size = sizeof(int8_t);
            break;
        case 8: // kTfLiteUInt8
            element_size = sizeof(uint8_t);
            break;
        case 2: // kTfLiteInt32
            element_size = sizeof(int32_t);
            break;
        case 7: // kTfLiteInt16
            element_size = sizeof(int16_t);
            break;
        case 3: // kTfLiteInt64
            element_size = sizeof(int64_t);
            break;
        default:
            MicroPrintf("ERRO: Tipo de tensor não suportado: %d", tensor->type);
            return;
    }
    
    element_count = tensor->bytes / element_size;
    
    if (index >= element_count) {
        MicroPrintf("ERRO: índice %zu >= %zu elementos", index, element_count);
        return;
    }
    
    // Processar baseado no tipo
    switch(tensor->type) {
        case 1: { // kTfLiteFloat32
            if (tensor->data.f == nullptr) {
                MicroPrintf("ERRO: tensor->data.f é NULL");
                return;
            }
            tensor->data.f[index] = value;
            MicroPrintf("Input[%zu] = %.6f", index, value);
            break;
        }
        
        case 2: { // kTfLiteInt32
            if (tensor->data.i32 == nullptr) {
                MicroPrintf("ERRO: tensor->data.i32 é NULL");
                return;
            }
            int32_t int_value = (int32_t)value;
            tensor->data.i32[index] = int_value;
            MicroPrintf("Input[%zu] = %d", index, int_value);
            break;
        }
        
        case 9: { // kTfLiteInt8
            if (tensor->data.int8 == nullptr) {
                MicroPrintf("ERRO: tensor->data.int8 é NULL");
                return;
            }
            
            // Para tensores quantizados, usar os parâmetros de quantização do tensor
            if (tensor->quantization.type == kTfLiteAffineQuantization) {
                const TfLiteAffineQuantization* quant = 
                    reinterpret_cast<const TfLiteAffineQuantization*>(tensor->quantization.params);
                
                if (quant && quant->scale && quant->zero_point && 
                    quant->scale->size > 0 && quant->zero_point->size > 0) {
                    
                    // Usar os parâmetros reais do tensor
                    float scale = quant->scale->data[0];
                    int32_t zero_point = quant->zero_point->data[0];
                    
                    // Quantização: q = round(value/scale) + zero_point
                    int32_t quantized = (int32_t)(value / scale + 0.5f) + zero_point;
                    
                    // Clamping para int8
                    if (quantized > 127) quantized = 127;
                    if (quantized < -128) quantized = -128;
                    
                    tensor->data.int8[index] = (int8_t)quantized;
                    MicroPrintf("Input[%zu] = %.6f (quantized to %d)", index, value, (int8_t)quantized);
                } else {
                    int8_t int_value = (int8_t)value;
                    if (int_value > 127) int_value = 127;
                    if (int_value < -128) int_value = -128;
                    tensor->data.int8[index] = int_value;
                    MicroPrintf("Input[%zu] = %d", index, int_value);
                }
            } else {
                int8_t int_value = (int8_t)value;
                if (int_value > 127) int_value = 127;
                if (int_value < -128) int_value = -128;
                tensor->data.int8[index] = int_value;
                MicroPrintf("Input[%zu] = %d", index, int_value);
            }
            break;
        }
        
        case 8: { // kTfLiteUInt8
            if (tensor->data.uint8 == nullptr) {
                MicroPrintf("ERRO: tensor->data.uint8 é NULL");
                return;
            }
            
            // Similar ao int8, mas para uint8
            if (tensor->quantization.type == kTfLiteAffineQuantization) {
                const TfLiteAffineQuantization* quant = 
                    reinterpret_cast<const TfLiteAffineQuantization*>(tensor->quantization.params);
                
                if (quant && quant->scale && quant->zero_point && 
                    quant->scale->size > 0 && quant->zero_point->size > 0) {
                    
                    float scale = quant->scale->data[0];
                    int32_t zero_point = quant->zero_point->data[0];
                    
                    int32_t quantized = (int32_t)(value / scale + 0.5f) + zero_point;
                    
                    // Clamping para uint8
                    if (quantized > 255) quantized = 255;
                    if (quantized < 0) quantized = 0;
                    
                    tensor->data.uint8[index] = (uint8_t)quantized;
                    MicroPrintf("Input[%zu] = %.6f (quantized to %u)", index, value, (uint8_t)quantized);
                } else {
                    uint8_t uint_value = (uint8_t)value;
                    if (value > 255.0f) uint_value = 255;
                    if (value < 0.0f) uint_value = 0;
                    tensor->data.uint8[index] = uint_value;
                    MicroPrintf("Input[%zu] = %u", index, uint_value);
                }
            } else {
                uint8_t uint_value = (uint8_t)value;
                if (value > 255.0f) uint_value = 255;
                if (value < 0.0f) uint_value = 0;
                tensor->data.uint8[index] = uint_value;
                MicroPrintf("Input[%zu] = %u", index, uint_value);
            }
            break;
        }
        
        case 7: { // kTfLiteInt16
            if (tensor->data.i16 == nullptr) {
                MicroPrintf("ERRO: tensor->data.i16 é NULL");
                return;
            }
            
            // Similar para int16
            if (tensor->quantization.type == kTfLiteAffineQuantization) {
                const TfLiteAffineQuantization* quant = 
                    reinterpret_cast<const TfLiteAffineQuantization*>(tensor->quantization.params);
                
                if (quant && quant->scale && quant->zero_point && 
                    quant->scale->size > 0 && quant->zero_point->size > 0) {
                    
                    float scale = quant->scale->data[0];
                    int32_t zero_point = quant->zero_point->data[0];
                    
                    int32_t quantized = (int32_t)(value / scale + 0.5f) + zero_point;
                    
                    // Clamping para int16
                    if (quantized > 32767) quantized = 32767;
                    if (quantized < -32768) quantized = -32768;
                    
                    tensor->data.i16[index] = (int16_t)quantized;
                    MicroPrintf("Input[%zu] = %.6f (quantized to %d)", index, value, (int16_t)quantized);
                } else {
                    int16_t int_value = (int16_t)value;
                    tensor->data.i16[index] = int_value;
                    MicroPrintf("Input[%zu] = %d", index, int_value);
                }
            } else {
                int16_t int_value = (int16_t)value;
                tensor->data.i16[index] = int_value;
                MicroPrintf("Input[%zu] = %d", index, int_value);
            }
            break;
        }
        
        case 3: { // kTfLiteInt64
            if (tensor->data.i64 == nullptr) {
                MicroPrintf("ERRO: tensor->data.i64 é NULL");
                return;
            }
            int64_t int_value = (int64_t)value;
            tensor->data.i64[index] = int_value;
            MicroPrintf("Input[%zu] = %lld", index, (long long)int_value);
            break;
        }
        
        default: {
            MicroPrintf("ERRO: Tipo de tensor não implementado: %d", tensor->type);
            break;
        }
    }
}

float GetTensorAsFloat(uintptr_t tensor_handle, size_t index, int) {
    if (tensor_handle == 0) {
        MicroPrintf("ERRO: tensor_handle nulo");
        return 0.0f;
    }
    
    const TfLiteTensor* tensor = reinterpret_cast<const TfLiteTensor*>(tensor_handle);
    if (tensor == nullptr) {
        MicroPrintf("ERRO: tensor nulo após cast");
        return 0.0f;
    }
    
    float result = 0.0f;
    
    // Processar baseado no tipo
    switch(tensor->type) {
        case 1: { // kTfLiteFloat32
            if (tensor->data.f == nullptr) {
                MicroPrintf("ERRO: tensor->data.f é NULL");
                return 0.0f;
            }
            size_t tensor_size = tensor->bytes / sizeof(float);
            if (index >= tensor_size) {
                MicroPrintf("ERRO: índice %zu >= %zu", index, tensor_size);
                return 0.0f;
            }
            result = tensor->data.f[index];
            MicroPrintf("Output[%zu] = %.6f", index, result);
            return result;
        }
        
        case 2: { // kTfLiteInt32
            if (tensor->data.i32 == nullptr) {
                MicroPrintf("ERRO: tensor->data.i32 é NULL");
                return 0.0f;
            }
            size_t tensor_size = tensor->bytes / sizeof(int32_t);
            if (index >= tensor_size) {
                MicroPrintf("ERRO: índice %zu >= %zu", index, tensor_size);
                return 0.0f;
            }
            result = (float)tensor->data.i32[index];
            MicroPrintf("Output[%zu] = %.6f (from int32: %d)", index, result, tensor->data.i32[index]);
            return result;
        }
        
        case 9: { // kTfLiteInt8
            if (tensor->data.int8 == nullptr) {
                MicroPrintf("ERRO: tensor->data.int8 é NULL");
                return 0.0f;
            }
            
            size_t element_size = sizeof(int8_t);
            size_t tensor_size = tensor->bytes / element_size;
            if (index >= tensor_size) {
                MicroPrintf("ERRO: índice %zu >= %zu", index, tensor_size);
                return 0.0f;
            }
            
            int8_t quantized_value = tensor->data.int8[index];
            
            // Para tensores quantizados, aplicar dequantização
            if (tensor->quantization.type == kTfLiteAffineQuantization) {
                const TfLiteAffineQuantization* quant = 
                    reinterpret_cast<const TfLiteAffineQuantization*>(tensor->quantization.params);
                
                if (quant && quant->scale && quant->zero_point && 
                    quant->scale->size > 0 && quant->zero_point->size > 0) {
                    
                    float scale = quant->scale->data[0];
                    int32_t zero_point = quant->zero_point->data[0];
                    
                    float dequantized = scale * (quantized_value - zero_point);
                    MicroPrintf("Output[%zu] = %.6f (dequantized from %d)", index, dequantized, quantized_value);
                    return dequantized;
                } else {
                    MicroPrintf("AVISO: Tensor int8 sem parâmetros de quantização válidos");
                }
            }
            
            result = (float)quantized_value;
            MicroPrintf("Output[%zu] = %.6f (from int8: %d)", index, result, quantized_value);
            return result;
        }
        
        case 8: { // kTfLiteUInt8
            if (tensor->data.uint8 == nullptr) {
                MicroPrintf("ERRO: tensor->data.uint8 é NULL");
                return 0.0f;
            }
            
            size_t tensor_size = tensor->bytes / sizeof(uint8_t);
            if (index >= tensor_size) {
                MicroPrintf("ERRO: índice %zu >= %zu", index, tensor_size);
                return 0.0f;
            }
            
            uint8_t quantized_value = tensor->data.uint8[index];
            
            // Para tensores quantizados, aplicar dequantização
            if (tensor->quantization.type == kTfLiteAffineQuantization) {
                const TfLiteAffineQuantization* quant = 
                    reinterpret_cast<const TfLiteAffineQuantization*>(tensor->quantization.params);
                
                if (quant && quant->scale && quant->zero_point && 
                    quant->scale->size > 0 && quant->zero_point->size > 0) {
                    
                    float scale = quant->scale->data[0];
                    int32_t zero_point = quant->zero_point->data[0];
                    
                    result = scale * (quantized_value - zero_point);
                    MicroPrintf("Output[%zu] = %.6f (dequantized from %u)", index, result, quantized_value);
                    return result;
                }
            }
            
            result = (float)quantized_value;
            MicroPrintf("Output[%zu] = %.6f (from uint8: %u)", index, result, quantized_value);
            return result;
        }
        
        default: {
            MicroPrintf("ERRO: Tipo não suportado: %d", tensor->type);
            return 0.0f;
        }
    }
}

size_t GetTensorSize(uintptr_t tensor_handle, int) {
    if (tensor_handle == 0) {
        MicroPrintf("ERRO: tensor_handle nulo");
        return 0;
    }
    
    const TfLiteTensor* tensor = reinterpret_cast<const TfLiteTensor*>(tensor_handle);
    if (tensor == nullptr) {
        MicroPrintf("ERRO: tensor nulo após cast");
        return 0;
    }
    
    size_t element_size;

    // Para tipo 0, assumir int32 como padrão
    if (tensor->type == kTfLiteNoType) {
        element_size = sizeof(int32_t);
    } else {
        // Calcula o número de elementos baseado no tipo
        switch(tensor->type) {
            case 1: // kTfLiteFloat32
                element_size = sizeof(float);
                break;
            case 9: // kTfLiteInt8
                element_size = sizeof(int8_t);
                break;
            case 8: // kTfLiteUInt8
                element_size = sizeof(uint8_t);
                break;
            case 2: // kTfLiteInt32
                element_size = sizeof(int32_t);
                break;
            case 7: // kTfLiteInt16
                element_size = sizeof(int16_t);
                break;
            case 3: // kTfLiteInt64
                element_size = sizeof(int64_t);
                break;
            default:
                MicroPrintf("ERRO: Tipo de tensor não suportado: %d", tensor->type);
                return 0;
        }
    }
    
    return tensor->bytes / element_size;
}

void SetTensorArray(uintptr_t tensor_handle, const float* values, size_t count, int) {
    if (tensor_handle == 0 || values == nullptr) {
        MicroPrintf("ERRO: tensor_handle ou values nulo");
        return;
    }
    
    size_t tensor_size = GetTensorSize(tensor_handle, 0);
    size_t copy_size = (count < tensor_size) ? count : tensor_size;
    
    MicroPrintf("Definindo %zu valores no tensor (tamanho: %zu)", copy_size, tensor_size);
    
    for (size_t i = 0; i < copy_size; ++i) {
        SetTensorValue(tensor_handle, i, values[i], 0);
    }
    
    // Preencher com zeros se necessário
    for (size_t i = copy_size; i < tensor_size; ++i) {
        SetTensorValue(tensor_handle, i, 0.0f, 0);
    }
}

// Função para obter um tensor completo como array
void GetTensorArray(uintptr_t tensor_handle, float* values, size_t max_count, int) {
    if (tensor_handle == 0 || values == nullptr) {
        MicroPrintf("ERRO: tensor_handle ou values nulo");
        return;
    }
    
    size_t tensor_size = GetTensorSize(tensor_handle, 0);
    size_t copy_size = (max_count < tensor_size) ? max_count : tensor_size;
    
    MicroPrintf("Obtendo %zu valores do tensor (tamanho: %zu)", copy_size, tensor_size);
    
    for (size_t i = 0; i < copy_size; ++i) {
        values[i] = GetTensorAsFloat(tensor_handle, i, 0);
    }
}

size_t GetInputTensorCount(uintptr_t instance_handle, int) {
    if (instance_handle == 0) return 0;
    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    return instance->interpreter->inputs_size();
}

size_t GetOutputTensorCount(uintptr_t instance_handle, int) {
    if (instance_handle == 0) return 0;
    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    return instance->interpreter->outputs_size();
}

void InvokeInterpreter(uintptr_t instance_handle, int) {
    if (instance_handle == 0) {
        MicroPrintf("ERRO: instance_handle nulo");
        return;
    }
    
    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    if (instance == nullptr || instance->interpreter == nullptr) {
        MicroPrintf("ERRO: instance ou interpreter nulo");
        return;
    }
    
    MicroPrintf("\n=== INICIANDO INFERÊNCIA ===");
    TfLiteStatus status = instance->interpreter->Invoke();
    
    if (status != kTfLiteOk) {
        MicroPrintf("ERRO na execução do interpreter: %d", status);
    } else {
        MicroPrintf("✓ Inferência executada com sucesso!");
    }
}

// Função de diagnóstico detalhado
void DiagnoseModel(const uint8_t* model_data, int) {
    MicroPrintf("\n=== DIAGNÓSTICO COMPLETO DO MODELO ===");
    
    const tflite::Model* model = tflite::GetModel(model_data);
    MicroPrintf("1. Model pointer: %p", model);
    MicroPrintf("2. Model version: %d (esperado: %d)", model->version(), TFLITE_SCHEMA_VERSION);
    
    if (!model->subgraphs()) {
        MicroPrintf("3. ERRO: Modelo sem subgrafos!");
        return;
    }
    MicroPrintf("3. Subgrafos: %zu", model->subgraphs()->size());
    
    const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
    if (!subgraph) {
        MicroPrintf("4. ERRO: Subgrafo 0 é NULL!");
        return;
    }
    
    MicroPrintf("4. Subgrafo 0 válido");
    MicroPrintf("5. Operadores: %zu", subgraph->operators() ? subgraph->operators()->size() : 0);
    MicroPrintf("6. Tensores: %zu", subgraph->tensors() ? subgraph->tensors()->size() : 0);
    MicroPrintf("7. Inputs: %zu", subgraph->inputs() ? subgraph->inputs()->size() : 0);
    MicroPrintf("8. Outputs: %zu", subgraph->outputs() ? subgraph->outputs()->size() : 0);
    
    // Analisar operadores
    if (subgraph->operators()) {
        for (size_t i = 0; i < subgraph->operators()->size(); ++i) {
            const tflite::Operator* op = subgraph->operators()->Get(i);
            if (!op) continue;
            
            uint32_t opcode_index = op->opcode_index();
            MicroPrintf("   Op[%zu]: opcode_index=%u", i, opcode_index);
        }
    }
    
    // Analisar opcodes
    const auto* opcodes = model->operator_codes();
    if (!opcodes) {
        MicroPrintf("9. ERRO: Modelo sem opcodes!");
        return;
    }
    MicroPrintf("9. Opcodes: %zu", opcodes->size());
    
    for (size_t i = 0; i < opcodes->size(); ++i) {
        const tflite::OperatorCode* opcode = opcodes->Get(i);
        if (!opcode) continue;
        
        tflite::BuiltinOperator builtin_code = opcode->builtin_code();
        int32_t deprecated_code = opcode->deprecated_builtin_code();
        
        MicroPrintf("   Opcode[%zu]: builtin=%d, deprecated=%d, name=%s", 
                   i, builtin_code, deprecated_code, 
                   tflite::EnumNameBuiltinOperator(builtin_code));
    }
    
    // Analisar tensores de input/output
    if (subgraph->tensors()) {
        for (size_t i = 0; i < subgraph->tensors()->size(); ++i) {
            const tflite::Tensor* tensor = subgraph->tensors()->Get(i);
            if (!tensor) continue;
            
            MicroPrintf("   Tensor[%zu]: type=%d", i, tensor->type());
        }
    }
}

// Função para verificar integridade do modelo
void VerifyModelData(const uint8_t* model_data, int) {
    MicroPrintf("\n=== VERIFICAÇÃO DE INTEGRIDADE DO MODELO ===");
    
    // Verificar magic number do TensorFlow Lite
    if (model_data[0] == 0x1c && model_data[1] == 0x0 && 
        model_data[2] == 0x0 && model_data[3] == 0x0 &&
        model_data[4] == 0x54 && model_data[5] == 0x46 &&
        model_data[6] == 0x4c && model_data[7] == 0x33) {
        MicroPrintf("✓ Magic number correto: TFL3");
    } else {
        MicroPrintf("✗ Magic number incorreto!");
        MicroPrintf("  Primeiros 8 bytes: %02x %02x %02x %02x %02x %02x %02x %02x",
                   model_data[0], model_data[1], model_data[2], model_data[3],
                   model_data[4], model_data[5], model_data[6], model_data[7]);
    }
    
    // Verificar ponteiro
    MicroPrintf("Ponteiro do modelo: %p", model_data);
    
    // Verificar se tflite::GetModel consegue parsear
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model) {
        MicroPrintf("✓ tflite::GetModel() funcionou");
        MicroPrintf("✓ Versão do modelo: %d", model->version());
    } else {
        MicroPrintf("✗ tflite::GetModel() retornou NULL!");
    }
}

// Função simples para imprimir float com quebras de linha
void PrintFloat(float value) {
    MicroPrintf("\n%.6f\n", value);
}

} // extern "C"