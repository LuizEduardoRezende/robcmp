#include "third-party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "third-party/tflite-micro/tensorflow/lite/schema/schema_generated.h"
#include "third-party/tflite-micro/tensorflow/lite/c/common.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/micro_log.h"
#include <cstring>
#include <cstdint>

// Incluir configuração de kernels
#include "kernel_config.h"


extern "C" {

// #define MicroPrintf(...) ((void)0)

#ifndef TFLM_MAX_OPS
#define TFLM_MAX_OPS 50
#endif

using MutableResolver = tflite::MicroMutableOpResolver<TFLM_MAX_OPS>;

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
  HASHTABLE_LOOKUP = 10, //Não suportado no TFLM
  L2_NORMALIZATION = 11,
  L2_POOL_2D = 12,
  LOCAL_RESPONSE_NORMALIZATION = 13, //Não suportado no TFLM
  LOGISTIC = 14,
  LSH_PROJECTION = 15, //Não suportado no TFLM
  LSTM = 16, //Não suportado no TFLM
  MAX_POOL_2D = 17,
  MUL = 18,
  RELU = 19,
  RELU_N1_TO_1 = 20, //Não suportado no TFLM
  RELU6 = 21,
  RESHAPE = 22,
  RESIZE_BILINEAR = 23,
  RNN = 24, //Não suportado no TFLM
  SOFTMAX = 25,
  SPACE_TO_DEPTH = 26,
  SVDF = 27,
  TANH = 28,
  CONCAT_EMBEDDINGS = 29, //Não suportado no TFLM
  SKIP_GRAM = 30, //Não suportado no TFLM
  CALL = 31, //Não suportado no TFLM
  CUSTOM = 32, //Não suportado no TFLM
  EMBEDDING_LOOKUP_SPARSE = 33, //Não suportado no TFLM
  PAD = 34,
  UNIDIRECTIONAL_SEQUENCE_RNN = 35, //Não suportado no TFLM
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
  BIDIRECTIONAL_SEQUENCE_RNN = 46, //Não suportado no TFLM
  EXP = 47,
  TOPK_V2 = 48, //Não suportado no TFLM
  SPLIT = 49,
  LOG_SOFTMAX = 50,
  DELEGATE = 51, //Não suportado no TFLM
  BIDIRECTIONAL_SEQUENCE_LSTM = 52, //Não suportado no TFLM
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
  SELECT = 64, //Não suportado no TFLM
  SLICE = 65,
  SIN = 66,
  TRANSPOSE_CONV = 67,
  SPARSE_TO_DENSE = 68, //Não suportado no TFLM
  TILE = 69, //Não suportado no TFLM
  EXPAND_DIMS = 70,
  EQUAL = 71,
  NOT_EQUAL = 72,
  LOG = 73,
  SUM = 74,
  SQRT = 75,
  RSQRT = 76,
  SHAPE = 77,
  POW = 78, //Não suportado no TFLM
  ARG_MIN = 79,
  FAKE_QUANT = 80, //Não suportado no TFLM
  REDUCE_PROD = 81, //Não suportado no TFLM
  REDUCE_MAX = 82,
  PACK = 83,
  LOGICAL_OR = 84,
  ONE_HOT = 85, //Não suportado no TFLM
  LOGICAL_AND = 86,
  LOGICAL_NOT = 87,
  UNPACK = 88,
  REDUCE_MIN = 89,
  FLOOR_DIV = 90,
  REDUCE_ANY = 91, //Não suportado no TFLM
  SQUARE = 92,
  ZEROS_LIKE = 93,
  FILL = 94,
  FLOOR_MOD = 95,
  RANGE = 96, //Não suportado no TFLM
  RESIZE_NEAREST_NEIGHBOR = 97,
  LEAKY_RELU = 98,
  SQUARED_DIFFERENCE = 99,
  MIRROR_PAD = 100,
  ABS = 101,
  SPLIT_V = 102,
  UNIQUE = 103, //Não suportado no TFLM
  CEIL = 104,
  REVERSE_V2 = 105,
  ADD_N = 106,
  GATHER_ND = 107,
  COS = 108,
  WHERE = 109, // Não suportado no TFLM
  RANK = 110, //Não suportado no TFLM
  ELU = 111,
  REVERSE_SEQUENCE = 112, //Não suportado no TFLM
  MATRIX_DIAG = 113, //Não suportado no TFLM
  QUANTIZE = 114,
  MATRIX_SET_DIAG = 115, //Não suportado no TFLM
  ROUND = 116,
  HARD_SWISH = 117,
  IF = 118,
  WHILE = 119,
  NON_MAX_SUPPRESSION_V4 = 120, //Não suportado no TFLM
  NON_MAX_SUPPRESSION_V5 = 121, //Não suportado no TFLM
  SCATTER_ND = 122, //Não suportado no TFLM
  SELECT_V2 = 123,
  DENSIFY = 124, //Não suportado no TFLM
  SEGMENT_SUM = 125, //Não suportado no TFLM
  BATCH_MATMUL = 126,
  PLACEHOLDER_FOR_GREATER_OP_CODES = 127, //Não suportado no TFLM
  CUMSUM = 128,
  CALL_ONCE = 129,
  BROADCAST_TO = 130,
  RFFT2D = 131, //Não suportado no TFLM
  CONV_3D = 132, //Não suportado no TFLM
  IMAG = 133, //Não suportado no TFLM
  REAL = 134, //Não suportado no TFLM
  COMPLEX_ABS = 135, //Não suportado no TFLM
  HASHTABLE = 136, //Não suportado no TFLM
  HASHTABLE_FIND = 137, //Não suportado no TFLM
  HASHTABLE_IMPORT = 138, //Não suportado no TFLM
  HASHTABLE_SIZE = 139, //Não suportado no TFLM
  REDUCE_ALL = 140, //Não suportado no TFLM
  CONV_3D_TRANSPOSE = 141, //Não suportado no TFLM
  VAR_HANDLE = 142,
  READ_VARIABLE = 143,
  ASSIGN_VARIABLE = 144,
  BROADCAST_ARGS = 145,
  RANDOM_STANDARD_NORMAL = 146,
  BUCKETIZE = 147, //Não suportado no TFLM
  RANDOM_UNIFORM = 148, //Não suportado no TFLM
  MULTINOMIAL = 149, //Não suportado no TFLM
  GELU = 150, //Não suportado no TFLM
  DYNAMIC_UPDATE_SLICE = 151, //Não suportado no TFLM
  RELU_0_TO_1 = 152, //Não suportado no TFLM
  UNSORTED_SEGMENT_PROD = 153, //Não suportado no TFLM
  UNSORTED_SEGMENT_MAX = 154, //Não suportado no TFLM
  UNSORTED_SEGMENT_SUM = 155, //Não suportado no TFLM
  ATAN2 = 156, //Não suportado no TFLM
  UNSORTED_SEGMENT_MIN = 157, //Não suportado no TFLM
  SIGN = 158, //Não suportado no TFLM
  BITCAST = 159, //Não suportado no TFLM
  BITWISE_XOR = 160, //Não suportado no TFLM
  RIGHT_SHIFT = 161, //Não suportado no TFLM
  STABLEHLO_LOGISTIC = 162, //Não suportado no TFLM
  STABLEHLO_ADD = 163, //Não suportado no TFLM
  STABLEHLO_DIVIDE = 164, //Não suportado no TFLM
  STABLEHLO_MULTIPLY = 165, //Não suportado no TFLM
  STABLEHLO_MAXIMUM = 166, //Não suportado no TFLM
  STABLEHLO_RESHAPE = 167, //Não suportado no TFLM
  STABLEHLO_CLAMP = 168, //Não suportado no TFLM
  STABLEHLO_CONCATENATE = 169, //Não suportado no TFLM
  STABLEHLO_BROADCAST_IN_DIM = 170, //Não suportado no TFLM
  STABLEHLO_CONVOLUTION = 171,//Não suportado no TFLM
  STABLEHLO_SLICE = 172, //Não suportado no TFLM
  STABLEHLO_CUSTOM_CALL = 173, //Não suportado no TFLM
  STABLEHLO_REDUCE = 174, //Não suportado no TFLM
  STABLEHLO_ABS = 175, //Não suportado no TFLM
  STABLEHLO_AND = 176, //Não suportado no TFLM
  STABLEHLO_COSINE = 177, //Não suportado no TFLM
  STABLEHLO_EXPONENTIAL = 178, //Não suportado no TFLM
  STABLEHLO_FLOOR = 179, //Não suportado no TFLM
  STABLEHLO_LOG = 180, //Não suportado no TFLM
  STABLEHLO_MINIMUM = 181, //Não suportado no TFLM
  STABLEHLO_NEGATE = 182, //Não suportado no TFLM
  STABLEHLO_OR = 183, //Não suportado no TFLM
  STABLEHLO_POWER = 184, //Não suportado no TFLM
  STABLEHLO_REMAINDER = 185, //Não suportado no TFLM
  STABLEHLO_RSQRT = 186, //Não suportado no TFLM
  STABLEHLO_SELECT = 187, //Não suportado no TFLM
  STABLEHLO_SUBTRACT = 188, //Não suportado no TFLM
  STABLEHLO_TANH = 189, //Não suportado no TFLM
  STABLEHLO_SCATTER = 190, //Não suportado no TFLM
  STABLEHLO_COMPARE = 191, //Não suportado no TFLM
  STABLEHLO_CONVERT = 192, //Não suportado no TFLM
  STABLEHLO_DYNAMIC_SLICE = 193, //Não suportado no TFLM
  STABLEHLO_DYNAMIC_UPDATE_SLICE = 194, //Não suportado no TFLM
  STABLEHLO_PAD = 195, //Não suportado no TFLM
  STABLEHLO_IOTA = 196, //Não suportado no TFLM
  STABLEHLO_DOT_GENERAL = 197, //Não suportado no TFLM
  STABLEHLO_REDUCE_WINDOW = 198, //Não suportado no TFLM
  STABLEHLO_SORT = 199, //Não suportado no TFLM
  STABLEHLO_WHILE = 200, //Não suportado no TFLM
  STABLEHLO_GATHER = 201, //Não suportado no TFLM
  STABLEHLO_TRANSPOSE = 202, //Não suportado no TFLM
  DILATE = 203, //Não suportado no TFLM
  STABLEHLO_RNG_BIT_GENERATOR = 204, //Não suportado no TFLM
  REDUCE_WINDOW = 205, //Não suportado no TFLM
  STABLEHLO_COMPOSITE = 206, //Não suportado no TFLM
  STABLEHLO_SHIFT_LEFT = 207, //Não suportado no TFLM
  STABLEHLO_CBRT = 208, //Não suportado no TFLM
  STABLEHLO_CASE = 209 //Não suportado no TFLM
} KernelType;

void RegisterOp(tflite::MicroMutableOpResolver<TFLM_MAX_OPS>* resolver, KernelType kernel_type);

struct TFLM_Instance {
    tflite::MicroInterpreter* interpreter;
    tflite::MicroAllocator* allocator;       
    MutableResolver* resolver;
};

void RegisterOp(tflite::MicroMutableOpResolver<TFLM_MAX_OPS>* resolver, KernelType kernel_type) {
    switch (kernel_type) {
        
#if ENABLE_ADD
        case ADD:
            resolver->AddAdd();
            MicroPrintf("✓ ADD registrado");
            break;
#endif

#if ENABLE_AVERAGE_POOL_2D
        case AVERAGE_POOL_2D:
            resolver->AddAveragePool2D();
            MicroPrintf("✓ AVERAGE_POOL_2D registrado");
            break;
#endif

#if ENABLE_CONCATENATION
        case CONCATENATION:
            resolver->AddConcatenation();
            MicroPrintf("✓ CONCATENATION registrado");
            break;
#endif

#if ENABLE_CONV_2D
        case CONV_2D:
            resolver->AddConv2D();
            MicroPrintf("✓ CONV_2D registrado");
            break;
#endif

#if ENABLE_DEPTHWISE_CONV_2D
        case DEPTHWISE_CONV_2D:
            resolver->AddDepthwiseConv2D();
            MicroPrintf("✓ DEPTHWISE_CONV_2D registrado");
            break;
#endif

#if ENABLE_DEPTH_TO_SPACE
        case DEPTH_TO_SPACE:
            resolver->AddDepthToSpace();
            MicroPrintf("✓ DEPTH_TO_SPACE registrado");
            break;
#endif

#if ENABLE_DEQUANTIZE
        case DEQUANTIZE:
            resolver->AddDequantize();
            MicroPrintf("✓ DEQUANTIZE registrado");
            break;
#endif

#if ENABLE_EMBEDDING_LOOKUP
        case EMBEDDING_LOOKUP:
            resolver->AddEmbeddingLookup();
            MicroPrintf("✓ EMBEDDING_LOOKUP registrado");
            break;
#endif

#if ENABLE_FLOOR
        case FLOOR:
            resolver->AddFloor();
            MicroPrintf("✓ FLOOR registrado");
            break;
#endif

#if ENABLE_FULLY_CONNECTED
        case FULLY_CONNECTED:
            resolver->AddFullyConnected();
            MicroPrintf("✓ FULLY_CONNECTED registrado");
            break;
#endif

#if ENABLE_L2_NORMALIZATION
        case L2_NORMALIZATION:
            resolver->AddL2Normalization();
            MicroPrintf("✓ L2_NORMALIZATION registrado");
            break;
#endif

#if ENABLE_L2_POOL_2D
        case L2_POOL_2D:
            resolver->AddL2Pool2D();
            MicroPrintf("✓ L2_POOL_2D registrado");
            break;
#endif

#if ENABLE_LOGISTIC
        case LOGISTIC:
            resolver->AddLogistic();
            MicroPrintf("✓ LOGISTIC registrado");
            break;
#endif

#if ENABLE_MAX_POOL_2D
        case MAX_POOL_2D:
            resolver->AddMaxPool2D();
            MicroPrintf("✓ MAX_POOL_2D registrado");
            break;
#endif

#if ENABLE_MUL
        case MUL:
            resolver->AddMul();
            MicroPrintf("✓ MUL registrado");
            break;
#endif

#if ENABLE_RELU
        case RELU:
            resolver->AddRelu();
            MicroPrintf("✓ RELU registrado");
            break;
#endif

#if ENABLE_RELU6
        case RELU6:
            resolver->AddRelu6();
            MicroPrintf("✓ RELU6 registrado");
            break;
#endif

#if ENABLE_RESHAPE
        case RESHAPE:
            resolver->AddReshape();
            MicroPrintf("✓ RESHAPE registrado");
            break;
#endif

#if ENABLE_RESIZE_BILINEAR
        case RESIZE_BILINEAR:
            resolver->AddResizeBilinear();
            MicroPrintf("✓ RESIZE_BILINEAR registrado");
            break;
#endif

#if ENABLE_SOFTMAX
        case SOFTMAX:
            resolver->AddSoftmax();
            MicroPrintf("✓ SOFTMAX registrado");
            break;
#endif

#if ENABLE_SPACE_TO_DEPTH
        case SPACE_TO_DEPTH:
            resolver->AddSpaceToDepth();
            MicroPrintf("✓ SPACE_TO_DEPTH registrado");
            break;
#endif

#if ENABLE_SVDF
        case SVDF:
            resolver->AddSvdf();
            MicroPrintf("✓ SVDF registrado");
            break;
#endif

#if ENABLE_TANH
        case TANH:
            resolver->AddTanh();
            MicroPrintf("✓ TANH registrado");
            break;
#endif

#if ENABLE_PAD
        case PAD:
            resolver->AddPad();
            MicroPrintf("✓ PAD registrado");
            break;
#endif

#if ENABLE_GATHER
        case GATHER:
            resolver->AddGather();
            MicroPrintf("✓ GATHER registrado");
            break;
#endif

#if ENABLE_BATCH_TO_SPACE_ND
        case BATCH_TO_SPACE_ND:
            resolver->AddBatchToSpaceNd();
            MicroPrintf("✓ BATCH_TO_SPACE_ND registrado");
            break;
#endif

#if ENABLE_SPACE_TO_BATCH_ND
        case SPACE_TO_BATCH_ND:
            resolver->AddSpaceToBatchNd();
            MicroPrintf("✓ SPACE_TO_BATCH_ND registrado");
            break;
#endif

#if ENABLE_TRANSPOSE
        case TRANSPOSE:
            resolver->AddTranspose();
            MicroPrintf("✓ TRANSPOSE registrado");
            break;
#endif

#if ENABLE_MEAN
        case MEAN:
            resolver->AddMean();
            MicroPrintf("✓ MEAN registrado");
            break;
#endif

#if ENABLE_SUB
        case SUB:
            resolver->AddSub();
            MicroPrintf("✓ SUB registrado");
            break;
#endif

#if ENABLE_DIV
        case DIV:
            resolver->AddDiv();
            MicroPrintf("✓ DIV registrado");
            break;
#endif

#if ENABLE_SQUEEZE
        case SQUEEZE:
            resolver->AddSqueeze();
            MicroPrintf("✓ SQUEEZE registrado");
            break;
#endif

#if ENABLE_UNIDIRECTIONAL_SEQUENCE_LSTM
        case UNIDIRECTIONAL_SEQUENCE_LSTM:
            resolver->AddUnidirectionalSequenceLSTM();
            MicroPrintf("✓ UNIDIRECTIONAL_SEQUENCE_LSTM registrado");
            break;
#endif

#if ENABLE_STRIDED_SLICE
        case STRIDED_SLICE:
            resolver->AddStridedSlice();
            MicroPrintf("✓ STRIDED_SLICE registrado");
            break;
#endif

#if ENABLE_EXP
        case EXP:
            resolver->AddExp();
            MicroPrintf("✓ EXP registrado");
            break;
#endif

#if ENABLE_SPLIT
        case SPLIT:
            resolver->AddSplit();
            MicroPrintf("✓ SPLIT registrado");
            break;
#endif

#if ENABLE_LOG_SOFTMAX
        case LOG_SOFTMAX:
            resolver->AddLogSoftmax();
            MicroPrintf("✓ LOG_SOFTMAX registrado");
            break;
#endif

#if ENABLE_CAST
        case CAST:
            resolver->AddCast();
            MicroPrintf("✓ CAST registrado");
            break;
#endif

#if ENABLE_PRELU
        case PRELU:
            resolver->AddPrelu();
            MicroPrintf("✓ PRELU registrado");
            break;
#endif

#if ENABLE_MAXIMUM
        case MAXIMUM:
            resolver->AddMaximum();
            MicroPrintf("✓ MAXIMUM registrado");
            break;
#endif

#if ENABLE_ARG_MAX
        case ARG_MAX:
            resolver->AddArgMax();
            MicroPrintf("✓ ARG_MAX registrado");
            break;
#endif

#if ENABLE_MINIMUM
        case MINIMUM:
            resolver->AddMinimum();
            MicroPrintf("✓ MINIMUM registrado");
            break;
#endif

#if ENABLE_LESS
        case LESS:
            resolver->AddLess();
            MicroPrintf("✓ LESS registrado");
            break;
#endif

#if ENABLE_NEG
        case NEG:
            resolver->AddNeg();
            MicroPrintf("✓ NEG registrado");
            break;
#endif

#if ENABLE_PADV2
        case PADV2:
            resolver->AddPadV2();
            MicroPrintf("✓ PADV2 registrado");
            break;
#endif

#if ENABLE_GREATER
        case GREATER:
            resolver->AddGreater();
            MicroPrintf("✓ GREATER registrado");
            break;
#endif

#if ENABLE_GREATER_EQUAL
        case GREATER_EQUAL:
            resolver->AddGreaterEqual();
            MicroPrintf("✓ GREATER_EQUAL registrado");
            break;
#endif

#if ENABLE_LESS_EQUAL
        case LESS_EQUAL:
            resolver->AddLessEqual();
            MicroPrintf("✓ LESS_EQUAL registrado");
            break;
#endif

#if ENABLE_SLICE
        case SLICE:
            resolver->AddSlice();
            MicroPrintf("✓ SLICE registrado");
            break;
#endif

#if ENABLE_SIN
        case SIN:
            resolver->AddSin();
            MicroPrintf("✓ SIN registrado");
            break;
#endif

#if ENABLE_TRANSPOSE_CONV
        case TRANSPOSE_CONV:
            resolver->AddTransposeConv();
            MicroPrintf("✓ TRANSPOSE_CONV registrado");
            break;
#endif

#if ENABLE_EXPAND_DIMS
        case EXPAND_DIMS:
            resolver->AddExpandDims();
            MicroPrintf("✓ EXPAND_DIMS registrado");
            break;
#endif

#if ENABLE_EQUAL
        case EQUAL:
            resolver->AddEqual();
            MicroPrintf("✓ EQUAL registrado");
            break;
#endif

#if ENABLE_NOT_EQUAL
        case NOT_EQUAL:
            resolver->AddNotEqual();
            MicroPrintf("✓ NOT_EQUAL registrado");
            break;
#endif

#if ENABLE_LOG
        case LOG:
            resolver->AddLog();
            MicroPrintf("✓ LOG registrado");
            break;
#endif

#if ENABLE_SUM
        case SUM:
            resolver->AddSum();
            MicroPrintf("✓ SUM registrado");
            break;
#endif

#if ENABLE_SQRT
        case SQRT:
            resolver->AddSqrt();
            MicroPrintf("✓ SQRT registrado");
            break;
#endif

#if ENABLE_RSQRT
        case RSQRT:
            resolver->AddRsqrt();
            MicroPrintf("✓ RSQRT registrado");
            break;
#endif

#if ENABLE_SHAPE
        case SHAPE:
            resolver->AddShape();
            MicroPrintf("✓ SHAPE registrado");
            break;
#endif

#if ENABLE_ARG_MIN
        case ARG_MIN:
            resolver->AddArgMin();
            MicroPrintf("✓ ARG_MIN registrado");
            break;
#endif

#if ENABLE_REDUCE_MAX
        case REDUCE_MAX:
            resolver->AddReduceMax();
            MicroPrintf("✓ REDUCE_MAX registrado");
            break;
#endif

#if ENABLE_PACK
        case PACK:
            resolver->AddPack();
            MicroPrintf("✓ PACK registrado");
            break;
#endif

#if ENABLE_LOGICAL_OR
        case LOGICAL_OR:
            resolver->AddLogicalOr();
            MicroPrintf("✓ LOGICAL_OR registrado");
            break;
#endif

#if ENABLE_LOGICAL_AND
        case LOGICAL_AND:
            resolver->AddLogicalAnd();
            MicroPrintf("✓ LOGICAL_AND registrado");
            break;
#endif

#if ENABLE_LOGICAL_NOT
        case LOGICAL_NOT:
            resolver->AddLogicalNot();
            MicroPrintf("✓ LOGICAL_NOT registrado");
            break;
#endif

#if ENABLE_UNPACK
        case UNPACK:
            resolver->AddUnpack();
            MicroPrintf("✓ UNPACK registrado");
            break;
#endif

#if ENABLE_REDUCE_MIN
        case REDUCE_MIN:
            resolver->AddReduceMin();
            MicroPrintf("✓ REDUCE_MIN registrado");
            break;
#endif

#if ENABLE_FLOOR_DIV
        case FLOOR_DIV:
            resolver->AddFloorDiv();
            MicroPrintf("✓ FLOOR_DIV registrado");
            break;
#endif

#if ENABLE_SQUARE
        case SQUARE:
            resolver->AddSquare();
            MicroPrintf("✓ SQUARE registrado");
            break;
#endif

#if ENABLE_ZEROS_LIKE
        case ZEROS_LIKE:
            resolver->AddZerosLike();
            MicroPrintf("✓ ZEROS_LIKE registrado");
            break;
#endif

#if ENABLE_FILL
        case FILL:
            resolver->AddFill();
            MicroPrintf("✓ FILL registrado");
            break;
#endif

#if ENABLE_FLOOR_MOD
        case FLOOR_MOD:
            resolver->AddFloorMod();
            MicroPrintf("✓ FLOOR_MOD registrado");
            break;
#endif

#if ENABLE_RESIZE_NEAREST_NEIGHBOR
        case RESIZE_NEAREST_NEIGHBOR:
            resolver->AddResizeNearestNeighbor();
            MicroPrintf("✓ RESIZE_NEAREST_NEIGHBOR registrado");
            break;
#endif

#if ENABLE_LEAKY_RELU
        case LEAKY_RELU:
            resolver->AddLeakyRelu();
            MicroPrintf("✓ LEAKY_RELU registrado");
            break;
#endif

#if ENABLE_SQUARED_DIFFERENCE
        case SQUARED_DIFFERENCE:
            resolver->AddSquaredDifference();
            MicroPrintf("✓ SQUARED_DIFFERENCE registrado");
            break;
#endif

#if ENABLE_MIRROR_PAD
        case MIRROR_PAD:
            resolver->AddMirrorPad();
            MicroPrintf("✓ MIRROR_PAD registrado");
            break;
#endif

#if ENABLE_ABS
        case ABS:
            resolver->AddAbs();
            MicroPrintf("✓ ABS registrado");
            break;
#endif

#if ENABLE_SPLIT_V
        case SPLIT_V:
            resolver->AddSplitV();
            MicroPrintf("✓ SPLIT_V registrado");
            break;
#endif

#if ENABLE_CEIL
        case CEIL:
            resolver->AddCeil();
            MicroPrintf("✓ CEIL registrado");
            break;
#endif

#if ENABLE_REVERSE_V2
        case REVERSE_V2:
            resolver->AddReverseV2();
            MicroPrintf("✓ REVERSE_V2 registrado");
            break;
#endif

#if ENABLE_ADD_N
        case ADD_N:
            resolver->AddAddN();
            MicroPrintf("✓ ADD_N registrado");
            break;
#endif

#if ENABLE_GATHER_ND
        case GATHER_ND:
            resolver->AddGatherNd();
            MicroPrintf("✓ GATHER_ND registrado");
            break;
#endif

#if ENABLE_COS
        case COS:
            resolver->AddCos();
            MicroPrintf("✓ COS registrado");
            break;
#endif

#if ENABLE_ELU
        case ELU:
            resolver->AddElu();
            MicroPrintf("✓ ELU registrado");
            break;
#endif

#if ENABLE_QUANTIZE
        case QUANTIZE:
            resolver->AddQuantize();
            MicroPrintf("✓ QUANTIZE registrado");
            break;
#endif

#if ENABLE_ROUND
        case ROUND:
            resolver->AddRound();
            MicroPrintf("✓ ROUND registrado");
            break;
#endif

#if ENABLE_HARD_SWISH
        case HARD_SWISH:
            resolver->AddHardSwish();
            MicroPrintf("✓ HARD_SWISH registrado");
            break;
#endif

#if ENABLE_IF
        case IF:
            resolver->AddIf();
            MicroPrintf("✓ IF registrado");
            break;
#endif

#if ENABLE_WHILE
        case WHILE:
            resolver->AddWhile();
            MicroPrintf("✓ WHILE registrado");
            break;
#endif

#if ENABLE_SELECT_V2
        case SELECT_V2:
            resolver->AddSelectV2();
            MicroPrintf("✓ SELECT_V2 registrado");
            break;
#endif

#if ENABLE_BATCH_MATMUL
        case BATCH_MATMUL:
            resolver->AddBatchMatMul();
            MicroPrintf("✓ BATCH_MATMUL registrado");
            break;
#endif

#if ENABLE_CUMSUM
        case CUMSUM:
            resolver->AddCumSum();
            MicroPrintf("✓ CUMSUM registrado");
            break;
#endif

#if ENABLE_CALL_ONCE
        case CALL_ONCE:
            resolver->AddCallOnce();
            MicroPrintf("✓ CALL_ONCE registrado");
            break;
#endif

#if ENABLE_BROADCAST_TO
        case BROADCAST_TO:
            resolver->AddBroadcastTo();
            MicroPrintf("✓ BROADCAST_TO registrado");
            break;
#endif

#if ENABLE_VAR_HANDLE
        case VAR_HANDLE:
            resolver->AddVarHandle();
            MicroPrintf("✓ VAR_HANDLE registrado");
            break;
#endif

#if ENABLE_READ_VARIABLE
        case READ_VARIABLE:
            resolver->AddReadVariable();
            MicroPrintf("✓ READ_VARIABLE registrado");
            break;
#endif

#if ENABLE_ASSIGN_VARIABLE
        case ASSIGN_VARIABLE:
            resolver->AddAssignVariable();
            MicroPrintf("✓ ASSIGN_VARIABLE registrado");
            break;
#endif

#if ENABLE_BROADCAST_ARGS
        case BROADCAST_ARGS:
            resolver->AddBroadcastArgs();
            MicroPrintf("✓ BROADCAST_ARGS registrado");
            break;
#endif
        default:
            MicroPrintf("!!! Kernel %d não suportado ou não habilitado na compilação", kernel_type);
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

    instance->allocator = tflite::MicroAllocator::Create(tensor_arena, tensor_arena_size);
    if (!instance->allocator) {
        MicroPrintf("ERRO: Falha ao criar MicroAllocator");
        delete instance->resolver;
        delete instance;
        return 0;
    }
    
    instance->interpreter = new tflite::MicroInterpreter(
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
            int32_t int_value = (int32_t)roundf(value); // arredondamento correto
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
                    int32_t quantized = (int32_t)roundf(value / scale) + zero_point;
                    
                    // Clamping para int8
                    if (quantized > 127) quantized = 127;
                    if (quantized < -128) quantized = -128;
                    
                    tensor->data.int8[index] = (int8_t)quantized;
                    MicroPrintf("Input[%zu] = %.6f (quantized to %d)", index, value, (int8_t)quantized);
                } else {
                    int8_t int_value = (int8_t)roundf(value);
                    if (int_value > 127) int_value = 127;
                    if (int_value < -128) int_value = -128;
                    tensor->data.int8[index] = int_value;
                    MicroPrintf("Input[%zu] = %d", index, int_value);
                }
            } else {
                int8_t int_value = (int8_t)roundf(value);
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
                    
                    int32_t quantized = (int32_t)roundf(value / scale) + zero_point;
                    
                    // Clamping para uint8
                    if (quantized > 255) quantized = 255;
                    if (quantized < 0) quantized = 0;
                    
                    tensor->data.uint8[index] = (uint8_t)quantized;
                    MicroPrintf("Input[%zu] = %.6f (quantized to %u)", index, value, (uint8_t)quantized);
                } else {
                    uint8_t uint_value = (uint8_t)roundf(value);
                    if (uint_value > 255) uint_value = 255;
                    if (uint_value < 0) uint_value = 0;
                    tensor->data.uint8[index] = uint_value;
                    MicroPrintf("Input[%zu] = %u", index, uint_value);
                }
            } else {
                uint8_t uint_value = (uint8_t)roundf(value);
                if (uint_value > 255) uint_value = 255;
                if (uint_value < 0) uint_value = 0;
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
                    
                    int32_t quantized = (int32_t)roundf(value / scale) + zero_point;
                    
                    // Clamping para int16
                    if (quantized > 32767) quantized = 32767;
                    if (quantized < -32768) quantized = -32768;
                    
                    tensor->data.i16[index] = (int16_t)quantized;
                    MicroPrintf("Input[%zu] = %.6f (quantized to %d)", index, value, (int16_t)quantized);
                } else {
                    int16_t int_value = (int16_t)roundf(value);
                    tensor->data.i16[index] = int_value;
                    MicroPrintf("Input[%zu] = %d", index, int_value);
                }
            } else {
                int16_t int_value = (int16_t)roundf(value);
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
            int64_t int_value = (int64_t)roundf(value);
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
            MicroPrintf("Output[%zu] = %.6f (float)", index, result);
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

void SetTensorFromIntArray(uintptr_t tensor_handle, const int16_t* values, size_t count, int) {
    if (tensor_handle == 0 || values == nullptr) {
        MicroPrintf("ERRO: SetTensorFromIntArray: handle ou valores nulos");
        return;
    }

    TfLiteTensor* tensor = reinterpret_cast<TfLiteTensor*>(tensor_handle);
    
    // Proteção contra escrita fora dos limites
    size_t num_elements = 0;
    if (tensor->type != kTfLiteNoType && tensor->bytes > 0) {
        // Calcula quantos elementos cabem no tensor baseado no tamanho do tipo
        size_t type_size = 1; // default
        switch(tensor->type) {
            case kTfLiteInt8: type_size = 1; break;
            case kTfLiteUInt8: type_size = 1; break;
            case kTfLiteInt16: type_size = 2; break;
            case kTfLiteInt32: type_size = 4; break;
            case kTfLiteFloat32: type_size = 4; break;
            default: type_size = 1; break;
        }
        num_elements = tensor->bytes / type_size;
    }

    if (count > num_elements) {
        MicroPrintf("AVISO: Tentativa de escrever %zu elementos em tensor de tamanho %zu. Truncando.", count, num_elements);
        count = num_elements;
    }

    // LOG DE DEBUG (Opcional: Mostra os primeiros dados recebidos)
    MicroPrintf("SetRaw: Copiando %zu elementos para tensor tipo %d", count, tensor->type);

    // Switch para tratar o destino corretamente
    switch (tensor->type) {
        case kTfLiteInt8: {
            int8_t* data = tflite::GetTensorData<int8_t>(tensor);
            for (size_t i = 0; i < count; ++i) {
                // Cast direto: int16 -> int8 (cuidado com overflow se o valor for > 127)
                data[i] = static_cast<int8_t>(values[i]);
            }
            break;
        }
        case kTfLiteUInt8: {
            uint8_t* data = tflite::GetTensorData<uint8_t>(tensor);
            for (size_t i = 0; i < count; ++i) {
                data[i] = static_cast<uint8_t>(values[i]);
            }
            break;
        }
        case kTfLiteInt16: {
            int16_t* data = tflite::GetTensorData<int16_t>(tensor);
            // Aqui podemos usar memcpy se a arquitetura permitir (endianness igual)
            // Mas o loop é mais seguro para garantir compatibilidade.
            for (size_t i = 0; i < count; ++i) {
                data[i] = values[i];
            }
            break;
        }
        case kTfLiteInt32: {
            int32_t* data = tflite::GetTensorData<int32_t>(tensor);
            for (size_t i = 0; i < count; ++i) {
                data[i] = static_cast<int32_t>(values[i]);
            }
            break;
        }
        case kTfLiteFloat32: {
            // Caso raro: O usuário quer passar inteiros para um tensor float sem conversão matemática
            float* data = tflite::GetTensorData<float>(tensor);
            for (size_t i = 0; i < count; ++i) {
                data[i] = static_cast<float>(values[i]);
            }
            break;
        }
        default:
            MicroPrintf("ERRO: Tipo de tensor %d não suportado para SetTensorFromIntArray", tensor->type);
            break;
    }
}

void GetTensorArray(uintptr_t tensor_handle, float* values, size_t max_count, int) {
    if (tensor_handle == 0 || values == nullptr) {
        MicroPrintf("ERRO: tensor_handle ou values nulo");
        return;
    }
    
    size_t tensor_size = GetTensorSize(tensor_handle, 0);
    size_t copy_size = (max_count < tensor_size) ? max_count : tensor_size;
    
    MicroPrintf("GetTensorArray: tensor_handle=%p, values(ptr)=%p, max_count=%zu, tensor_size=%zu, copy_size=%zu", (void*)tensor_handle, (void*)values, max_count, tensor_size, copy_size);
    
    for (size_t i = 0; i < copy_size; ++i) {
        float v = GetTensorAsFloat(tensor_handle, i, 0);
        values[i] = v;
        MicroPrintf("GetTensorArray: values[%zu]=%f (copied)", i, v);
    }
}

// Função para alocar e retornar um array preenchido com os valores do tensor
float* AllocAndGetTensorArray(uintptr_t tensor_handle, size_t* out_size, int) {
    size_t tensor_size = GetTensorSize(tensor_handle, 0);
    float* arr = (float*)malloc(sizeof(float) * tensor_size);
    if (!arr) {
        MicroPrintf("ERRO: malloc falhou em AllocAndGetTensorArray");
        if (out_size) *out_size = 0;
        return nullptr;
    }
    GetTensorArray(tensor_handle, arr, tensor_size, 0);
    if (out_size) *out_size = tensor_size;
    return arr;
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