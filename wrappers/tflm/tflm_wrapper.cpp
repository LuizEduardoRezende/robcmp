#include "third-party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "third-party/tflite-micro/tensorflow/lite/schema/schema_generated.h"
#include "third-party/tflite-micro/tensorflow/lite/c/common.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/micro_log.h"
#include <cstring>

extern "C" {

using MutableResolver = tflite::MicroMutableOpResolver<100>;

// Enumeração para tipos de kernel suportados
typedef enum {
    KERNEL_ABS = 0,
    KERNEL_ADD = 1,
    KERNEL_ADD_N = 2,
    KERNEL_ARG_MAX = 3,
    KERNEL_ARG_MIN = 4,
    KERNEL_ASSIGN_VARIABLE = 5,
    KERNEL_AVERAGE_POOL_2D = 6,
    KERNEL_BATCH_MATMUL = 7,
    KERNEL_BATCH_TO_SPACE_ND = 8,
    KERNEL_BROADCAST_ARGS = 9,
    KERNEL_BROADCAST_TO = 10,
    KERNEL_CALL_ONCE = 11,
    KERNEL_CAST = 12,
    KERNEL_CEIL = 13,
    // TODO(b/160234179): Change custom OPs to also return by value.
    KERNEL_CIRCULAR_BUFFER = 14,
    KERNEL_CONCATENATION = 15,
    KERNEL_CONV_2D = 16,
    KERNEL_COS = 17,
    KERNEL_CUMSUM = 18,
    KERNEL_DEPTH_TO_SPACE = 19,
    KERNEL_DEPTHWISE_CONV_2D = 20,
    KERNEL_DEQUANTIZE = 21,
    KERNEL_DIV = 22,
    KERNEL_ELU = 23,
    KERNEL_EMBEDDING_LOOKUP = 24,
    KERNEL_EQUAL = 25,
    KERNEL_ETHOSU = 26,
    KERNEL_EXP = 27,
    KERNEL_EXPAND_DIMS = 28,
    KERNEL_FILL = 29,
    KERNEL_FLOOR = 30,
    KERNEL_FLOOR_DIV = 31,
    KERNEL_FLOOR_MOD = 32,
    KERNEL_FULLY_CONNECTED = 33,
    KERNEL_GATHER = 34,
    KERNEL_GATHER_ND = 35,
    KERNEL_GREATER = 36,
    KERNEL_GREATER_EQUAL = 37,
    KERNEL_HARD_SWISH = 38,
    KERNEL_IF = 39,
    KERNEL_L2_NORMALIZATION = 40,
    KERNEL_L2_POOL_2D = 41,
    KERNEL_LEAKY_RELU = 42,
    KERNEL_LESS = 43,
    KERNEL_LESS_EQUAL = 44,
    KERNEL_LOG = 45,
    KERNEL_LOG_SOFTMAX = 46,
    KERNEL_LOGICAL_AND = 47,
    KERNEL_LOGICAL_NOT = 48,
    KERNEL_LOGICAL_OR = 49,
    KERNEL_LOGISTIC = 50,
    KERNEL_MAX_POOL_2D = 51,
    KERNEL_MAXIMUM = 52,
    KERNEL_MEAN = 53,
    KERNEL_MINIMUM = 54,
    KERNEL_MIRROR_PAD = 55,
    KERNEL_MUL = 56,
    KERNEL_NEG = 57,
    KERNEL_NOT_EQUAL = 58,
    KERNEL_PACK = 59,
    KERNEL_PAD = 60,
    KERNEL_PADV2 = 61,
    KERNEL_PRELU = 62,
    KERNEL_QUANTIZE = 63,
    KERNEL_READ_VARIABLE = 64,
    KERNEL_REDUCE_MAX = 65,
    KERNEL_REDUCE_MIN = 66,
    KERNEL_RELU = 67,
    KERNEL_RELU6 = 68,
    KERNEL_RESHAPE = 69,
    KERNEL_RESIZE_BILINEAR = 70,
    KERNEL_RESIZE_NEAREST_NEIGHBOR = 71,
    KERNEL_REVERSE_V2 = 72,
    KERNEL_ROUND = 73,
    KERNEL_RSQRT = 74,
    KERNEL_SELECT_V2 = 75,
    KERNEL_SHAPE = 76,
    KERNEL_SIN = 77,
    KERNEL_SLICE = 78,
    KERNEL_SOFTMAX = 79,
    KERNEL_SPACE_TO_BATCH_ND = 80,
    KERNEL_SPACE_TO_DEPTH = 81,
    KERNEL_SPLIT = 82,
    KERNEL_SPLIT_V = 83,
    KERNEL_SQRT = 84,
    KERNEL_SQUARE = 85,
    KERNEL_SQUARED_DIFFERENCE = 86,
    KERNEL_SQUEEZE = 87,
    KERNEL_STRIDED_SLICE = 88,
    KERNEL_SUB = 89,
    KERNEL_SUM = 90,
    KERNEL_SVDF = 91,
    KERNEL_TANH = 92,
    KERNEL_TRANSPOSE = 93,
    KERNEL_TRANSPOSE_CONV = 94,
    // TODO(b/230666079): resolve conflict with xtensa implementation
    KERNEL_UNIDIRECTIONAL_SEQUENCE_LSTM = 95,
    KERNEL_UNPACK = 96,
    KERNEL_VAR_HANDLE = 97,
    KERNEL_WHILE = 98,
    KERNEL_ZEROS_LIKE = 99
} KernelType;

void RegisterOp(tflite::MicroMutableOpResolver<100>* resolver, KernelType kernel_type);

struct TFLM_Instance {
    tflite::MicroInterpreter* interpreter;
    MutableResolver* resolver;
};

void* InitializeInterpreter(const uint8_t* model_data, uint8_t* tensor_arena, size_t tensor_arena_size, const uint8_t* required_kernels, size_t num_kernels) {    
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema version mismatch!");
        return nullptr;
    }

    TFLM_Instance* instance = new TFLM_Instance();
    instance->resolver = new MutableResolver();

    // Registrar kernels convertendo uint8_t para KernelType
    MicroPrintf("Registrando %zu kernels necessários:", num_kernels);
    for (size_t i = 0; i < num_kernels; ++i) {
        KernelType kernel_type = static_cast<KernelType>(required_kernels[i]);
        RegisterOp(instance->resolver, kernel_type);
    }
    
    instance->interpreter = new tflite::MicroInterpreter(model, *(instance->resolver), tensor_arena, tensor_arena_size);

    if (instance->interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("Failed to allocate tensors! Verifique se todas as ops foram registradas.");
        delete instance->interpreter;
        delete instance->resolver;
        delete instance;
        return nullptr;
    }

    MicroPrintf("Interpreter initialized successfully.");
    return reinterpret_cast<void*>(instance);
}

void DestroyInterpreter(void* instance_handle) {
    if (instance_handle == nullptr) return;

    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    delete instance->interpreter;
    delete instance->resolver; 
    delete instance;
    MicroPrintf("Interpreter destroyed successfully.");
}

TfLiteTensor* GetInputTensor(void* instance_handle, size_t index) {
    if (instance_handle == nullptr) return nullptr;
    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    return instance->interpreter->input(index);
}

const TfLiteTensor* GetOutputTensor(void* instance_handle, size_t index) {
    if (instance_handle == nullptr) return nullptr;
    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    return instance->interpreter->output(index);
}

void InvokeInterpreter(void* instance_handle) {
    if (instance_handle == nullptr) {
        MicroPrintf("Interpreter pointer is null.");
        return;
    }

    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    if (instance->interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Failed to invoke interpreter.");
    } else {
        MicroPrintf("Interpreter invoked successfully.");
    }
}

void RegisterOp(tflite::MicroMutableOpResolver<100>* resolver, KernelType kernel_type) {
    switch (kernel_type) {
        case KERNEL_ABS:
            resolver->AddAbs();
            MicroPrintf("Registrado kernel: ABS");
            break;
        case KERNEL_ADD:
            resolver->AddAdd();
            MicroPrintf("Registrado kernel: ADD");
            break;
        case KERNEL_ADD_N:
            resolver->AddAddN();
            MicroPrintf("Registrado kernel: ADD_N");
            break;
        case KERNEL_ARG_MAX:
            resolver->AddArgMax();
            MicroPrintf("Registrado kernel: ARG_MAX");
            break;
        case KERNEL_ARG_MIN:
            resolver->AddArgMin();
            MicroPrintf("Registrado kernel: ARG_MIN");
            break;
        case KERNEL_ASSIGN_VARIABLE:
            resolver->AddAssignVariable();
            MicroPrintf("Registrado kernel: ASSIGN_VARIABLE");
            break;
        case KERNEL_AVERAGE_POOL_2D:
            resolver->AddAveragePool2D();
            MicroPrintf("Registrado kernel: AVERAGE_POOL_2D");
            break;
        case KERNEL_BATCH_MATMUL:
            resolver->AddBatchMatMul();
            MicroPrintf("Registrado kernel: BATCH_MATMUL");
            break;
        case KERNEL_BATCH_TO_SPACE_ND:
            resolver->AddBatchToSpaceNd();
            MicroPrintf("Registrado kernel: BATCH_TO_SPACE_ND");
            break;
        case KERNEL_BROADCAST_ARGS:
            resolver->AddBroadcastArgs();
            MicroPrintf("Registrado kernel: BROADCAST_ARGS");
            break;
        case KERNEL_BROADCAST_TO:
            resolver->AddBroadcastTo();
            MicroPrintf("Registrado kernel: BROADCAST_TO");
            break;
        case KERNEL_CALL_ONCE:
            resolver->AddCallOnce();
            MicroPrintf("Registrado kernel: CALL_ONCE");
            break;
        case KERNEL_CAST:
            resolver->AddCast();
            MicroPrintf("Registrado kernel: CAST");
            break;
        case KERNEL_CEIL:
            resolver->AddCeil();
            MicroPrintf("Registrado kernel: CEIL");
            break;
        case KERNEL_CIRCULAR_BUFFER:
            resolver->AddCircularBuffer();
            MicroPrintf("Registrado kernel: CIRCULAR_BUFFER");
            break;
        case KERNEL_CONCATENATION:
            resolver->AddConcatenation();
            MicroPrintf("Registrado kernel: CONCATENATION");
            break;
        case KERNEL_CONV_2D:
            resolver->AddConv2D();
            MicroPrintf("Registrado kernel: CONV_2D");
            break;
        case KERNEL_COS:
            resolver->AddCos();
            MicroPrintf("Registrado kernel: COS");
            break;
        case KERNEL_CUMSUM:
            resolver->AddCumSum();
            MicroPrintf("Registrado kernel: CUMSUM");
            break;
        case KERNEL_DEPTH_TO_SPACE:
            resolver->AddDepthToSpace();
            MicroPrintf("Registrado kernel: DEPTH_TO_SPACE");
            break;
        case KERNEL_DEPTHWISE_CONV_2D:
            resolver->AddDepthwiseConv2D();
            MicroPrintf("Registrado kernel: DEPTHWISE_CONV_2D");
            break;
        case KERNEL_DEQUANTIZE:
            resolver->AddDequantize();
            MicroPrintf("Registrado kernel: DEQUANTIZE");
            break;
        case KERNEL_DIV:
            resolver->AddDiv();
            MicroPrintf("Registrado kernel: DIV");
            break;
        case KERNEL_ELU:
            resolver->AddElu();
            MicroPrintf("Registrado kernel: ELU");
            break;
        case KERNEL_EMBEDDING_LOOKUP:
            resolver->AddEmbeddingLookup();
            MicroPrintf("Registrado kernel: EMBEDDING_LOOKUP");
            break;
        case KERNEL_EQUAL:
            resolver->AddEqual();
            MicroPrintf("Registrado kernel: EQUAL");
            break;
        case KERNEL_ETHOSU:
            resolver->AddEthosU();
            MicroPrintf("Registrado kernel: ETHOSU");
            break;
        case KERNEL_EXP:
            resolver->AddExp();
            MicroPrintf("Registrado kernel: EXP");
            break;
        case KERNEL_EXPAND_DIMS:
            resolver->AddExpandDims();
            MicroPrintf("Registrado kernel: EXPAND_DIMS");
            break;
        case KERNEL_FILL:
            resolver->AddFill();
            MicroPrintf("Registrado kernel: FILL");
            break;
        case KERNEL_FLOOR:
            resolver->AddFloor();
            MicroPrintf("Registrado kernel: FLOOR");
            break;
        case KERNEL_FLOOR_DIV:
            resolver->AddFloorDiv();
            MicroPrintf("Registrado kernel: FLOOR_DIV");
            break;
        case KERNEL_FLOOR_MOD:
            resolver->AddFloorMod();
            MicroPrintf("Registrado kernel: FLOOR_MOD");
            break;
        case KERNEL_FULLY_CONNECTED:
            resolver->AddFullyConnected();
            MicroPrintf("Registrado kernel: FULLY_CONNECTED");
            break;
        case KERNEL_GATHER:
            resolver->AddGather();
            MicroPrintf("Registrado kernel: GATHER");
            break;
        case KERNEL_GATHER_ND:
            resolver->AddGatherNd();
            MicroPrintf("Registrado kernel: GATHER_ND");
            break;
        case KERNEL_GREATER:
            resolver->AddGreater();
            MicroPrintf("Registrado kernel: GREATER");
            break;
        case KERNEL_GREATER_EQUAL:
            resolver->AddGreaterEqual();
            MicroPrintf("Registrado kernel: GREATER_EQUAL");
            break;
        case KERNEL_HARD_SWISH:
            resolver->AddHardSwish();
            MicroPrintf("Registrado kernel: HARD_SWISH");
            break;
        case KERNEL_IF:
            resolver->AddIf();
            MicroPrintf("Registrado kernel: IF");
            break;
        case KERNEL_L2_NORMALIZATION:
            resolver->AddL2Normalization();
            MicroPrintf("Registrado kernel: L2_NORMALIZATION");
            break;
        case KERNEL_L2_POOL_2D:
            resolver->AddL2Pool2D();
            MicroPrintf("Registrado kernel: L2_POOL_2D");
            break;
        case KERNEL_LEAKY_RELU:
            resolver->AddLeakyRelu();
            MicroPrintf("Registrado kernel: LEAKY_RELU");
            break;
        case KERNEL_LESS:
            resolver->AddLess();
            MicroPrintf("Registrado kernel: LESS");
            break;
        case KERNEL_LESS_EQUAL:
            resolver->AddLessEqual();
            MicroPrintf("Registrado kernel: LESS_EQUAL");
            break;
        case KERNEL_LOG:
            resolver->AddLog();
            MicroPrintf("Registrado kernel: LOG");
            break;
        case KERNEL_LOG_SOFTMAX:
            resolver->AddLogSoftmax();
            MicroPrintf("Registrado kernel: LOG_SOFTMAX");
            break;
        case KERNEL_LOGICAL_AND:
            resolver->AddLogicalAnd();
            MicroPrintf("Registrado kernel: LOGICAL_AND");
            break;
        case KERNEL_LOGICAL_NOT:
            resolver->AddLogicalNot();
            MicroPrintf("Registrado kernel: LOGICAL_NOT");
            break;
        case KERNEL_LOGICAL_OR:
            resolver->AddLogicalOr();
            MicroPrintf("Registrado kernel: LOGICAL_OR");
            break;
        case KERNEL_LOGISTIC:
            resolver->AddLogistic();
            MicroPrintf("Registrado kernel: LOGISTIC");
            break;
        case KERNEL_MAX_POOL_2D:
            resolver->AddMaxPool2D();
            MicroPrintf("Registrado kernel: MAX_POOL_2D");
            break;
        case KERNEL_MAXIMUM:
            resolver->AddMaximum();
            MicroPrintf("Registrado kernel: MAXIMUM");
            break;
        case KERNEL_MEAN:
            resolver->AddMean();
            MicroPrintf("Registrado kernel: MEAN");
            break;
        case KERNEL_MINIMUM:
            resolver->AddMinimum();
            MicroPrintf("Registrado kernel: MINIMUM");
            break;
        case KERNEL_MIRROR_PAD:
            resolver->AddMirrorPad();
            MicroPrintf("Registrado kernel: MIRROR_PAD");
            break;
        case KERNEL_MUL:
            resolver->AddMul();
            MicroPrintf("Registrado kernel: MUL");
            break;
        case KERNEL_NEG:
            resolver->AddNeg();
            MicroPrintf("Registrado kernel: NEG");
            break;
        case KERNEL_NOT_EQUAL:
            resolver->AddNotEqual();
            MicroPrintf("Registrado kernel: NOT_EQUAL");
            break;
        case KERNEL_PACK:
            resolver->AddPack();
            MicroPrintf("Registrado kernel: PACK");
            break;
        case KERNEL_PAD:
            resolver->AddPad();
            MicroPrintf("Registrado kernel: PAD");
            break;
        case KERNEL_PADV2:
            resolver->AddPadV2();
            MicroPrintf("Registrado kernel: PADV2");
            break;
        case KERNEL_PRELU:
            resolver->AddPrelu();
            MicroPrintf("Registrado kernel: PRELU");
            break;
        case KERNEL_QUANTIZE:
            resolver->AddQuantize();
            MicroPrintf("Registrado kernel: QUANTIZE");
            break;
        case KERNEL_READ_VARIABLE:
            resolver->AddReadVariable();
            MicroPrintf("Registrado kernel: READ_VARIABLE");
            break;
        case KERNEL_REDUCE_MAX:
            resolver->AddReduceMax();
            MicroPrintf("Registrado kernel: REDUCE_MAX");
            break;
        case KERNEL_REDUCE_MIN:
            resolver->AddReduceMin();
            MicroPrintf("Registrado kernel: REDUCE_MIN");
            break;
        case KERNEL_RELU:
            resolver->AddRelu();
            MicroPrintf("Registrado kernel: RELU");
            break;
        case KERNEL_RELU6:
            resolver->AddRelu6();
            MicroPrintf("Registrado kernel: RELU6");
            break;
        case KERNEL_RESHAPE:
            resolver->AddReshape();
            MicroPrintf("Registrado kernel: RESHAPE");
            break;
        case KERNEL_RESIZE_BILINEAR:
            resolver->AddResizeBilinear();
            MicroPrintf("Registrado kernel: RESIZE_BILINEAR");
            break;
        case KERNEL_RESIZE_NEAREST_NEIGHBOR:
            resolver->AddResizeNearestNeighbor();
            MicroPrintf("Registrado kernel: RESIZE_NEAREST_NEIGHBOR");
            break;
        case KERNEL_REVERSE_V2:
            resolver->AddReverseV2();
            MicroPrintf("Registrado kernel: REVERSE_V2");
            break;
        case KERNEL_ROUND:
            resolver->AddRound();
            MicroPrintf("Registrado kernel: ROUND");
            break;
        case KERNEL_RSQRT:
            resolver->AddRsqrt();
            MicroPrintf("Registrado kernel: RSQRT");
            break;
        case KERNEL_SELECT_V2:
            resolver->AddSelectV2();
            MicroPrintf("Registrado kernel: SELECT_V2");
            break;
        case KERNEL_SHAPE:
            resolver->AddShape();
            MicroPrintf("Registrado kernel: SHAPE");
            break;
        case KERNEL_SIN:
            resolver->AddSin();
            MicroPrintf("Registrado kernel: SIN");
            break;
        case KERNEL_SLICE:
            resolver->AddSlice();
            MicroPrintf("Registrado kernel: SLICE");
            break;
        case KERNEL_SOFTMAX:
            resolver->AddSoftmax();
            MicroPrintf("Registrado kernel: SOFTMAX");
            break;
        case KERNEL_SPACE_TO_BATCH_ND:
            resolver->AddSpaceToBatchNd();
            MicroPrintf("Registrado kernel: SPACE_TO_BATCH_ND");
            break;
        case KERNEL_SPACE_TO_DEPTH:
            resolver->AddSpaceToDepth();
            MicroPrintf("Registrado kernel: SPACE_TO_DEPTH");
            break;
        case KERNEL_SPLIT:
            resolver->AddSplit();
            MicroPrintf("Registrado kernel: SPLIT");
            break;
        case KERNEL_SPLIT_V:
            resolver->AddSplitV();
            MicroPrintf("Registrado kernel: SPLIT_V");
            break;
        case KERNEL_SQRT:
            resolver->AddSqrt();
            MicroPrintf("Registrado kernel: SQRT");
            break;
        case KERNEL_SQUARE:
            resolver->AddSquare();
            MicroPrintf("Registrado kernel: SQUARE");
            break;
        case KERNEL_SQUARED_DIFFERENCE:
            resolver->AddSquaredDifference();
            MicroPrintf("Registrado kernel: SQUARED_DIFFERENCE");
            break;
        case KERNEL_SQUEEZE:
            resolver->AddSqueeze();
            MicroPrintf("Registrado kernel: SQUEEZE");
            break;
        case KERNEL_STRIDED_SLICE:
            resolver->AddStridedSlice();
            MicroPrintf("Registrado kernel: STRIDED_SLICE");
            break;
        case KERNEL_SUB:
            resolver->AddSub();
            MicroPrintf("Registrado kernel: SUB");
            break;
        case KERNEL_SUM:
            resolver->AddSum();
            MicroPrintf("Registrado kernel: SUM");
            break;
        case KERNEL_SVDF:
            resolver->AddSvdf();
            MicroPrintf("Registrado kernel: SVDF");
            break;
        case KERNEL_TANH:
            resolver->AddTanh();
            MicroPrintf("Registrado kernel: TANH");
            break;
        case KERNEL_TRANSPOSE:
            resolver->AddTranspose();
            MicroPrintf("Registrado kernel: TRANSPOSE");
            break;
        case KERNEL_TRANSPOSE_CONV:
            resolver->AddTransposeConv();
            MicroPrintf("Registrado kernel: TRANSPOSE_CONV");
            break;
        case KERNEL_UNIDIRECTIONAL_SEQUENCE_LSTM:
            resolver->AddUnidirectionalSequenceLSTM();
            MicroPrintf("Registrado kernel: UNIDIRECTIONAL_SEQUENCE_LSTM");
            break;
        case KERNEL_UNPACK:
            resolver->AddUnpack();
            MicroPrintf("Registrado kernel: UNPACK");
            break;
        case KERNEL_VAR_HANDLE:
            resolver->AddVarHandle();
            MicroPrintf("Registrado kernel: VAR_HANDLE");
            break;
        case KERNEL_WHILE:
            resolver->AddWhile();
            MicroPrintf("Registrado kernel: WHILE");
            break;
        case KERNEL_ZEROS_LIKE:
            resolver->AddZerosLike();
            MicroPrintf("Registrado kernel: ZEROS_LIKE");
            break;
        default:
            MicroPrintf("Kernel type não suportado: %d", kernel_type);
            break;
    }
}

KernelType MapBuiltinOperatorToKernelType(tflite::BuiltinOperator builtin_op) {
    switch (builtin_op) {
        case tflite::BuiltinOperator_ABS: return KERNEL_ABS;
        case tflite::BuiltinOperator_ADD: return KERNEL_ADD;
        case tflite::BuiltinOperator_ADD_N: return KERNEL_ADD_N;
        case tflite::BuiltinOperator_ARG_MAX: return KERNEL_ARG_MAX;
        case tflite::BuiltinOperator_ARG_MIN: return KERNEL_ARG_MIN;
        case tflite::BuiltinOperator_AVERAGE_POOL_2D: return KERNEL_AVERAGE_POOL_2D;
        case tflite::BuiltinOperator_BATCH_MATMUL: return KERNEL_BATCH_MATMUL;
        case tflite::BuiltinOperator_BATCH_TO_SPACE_ND: return KERNEL_BATCH_TO_SPACE_ND;
        case tflite::BuiltinOperator_CAST: return KERNEL_CAST;
        case tflite::BuiltinOperator_CEIL: return KERNEL_CEIL;
        case tflite::BuiltinOperator_CONCATENATION: return KERNEL_CONCATENATION;
        case tflite::BuiltinOperator_CONV_2D: return KERNEL_CONV_2D;
        case tflite::BuiltinOperator_COS: return KERNEL_COS;
        case tflite::BuiltinOperator_DEPTHWISE_CONV_2D: return KERNEL_DEPTHWISE_CONV_2D;
        case tflite::BuiltinOperator_DEQUANTIZE: return KERNEL_DEQUANTIZE;
        case tflite::BuiltinOperator_DIV: return KERNEL_DIV;
        case tflite::BuiltinOperator_ELU: return KERNEL_ELU;
        case tflite::BuiltinOperator_EQUAL: return KERNEL_EQUAL;
        case tflite::BuiltinOperator_EXP: return KERNEL_EXP;
        case tflite::BuiltinOperator_EXPAND_DIMS: return KERNEL_EXPAND_DIMS;
        case tflite::BuiltinOperator_FILL: return KERNEL_FILL;
        case tflite::BuiltinOperator_FLOOR: return KERNEL_FLOOR;
        case tflite::BuiltinOperator_FLOOR_DIV: return KERNEL_FLOOR_DIV;
        case tflite::BuiltinOperator_FLOOR_MOD: return KERNEL_FLOOR_MOD;
        case tflite::BuiltinOperator_FULLY_CONNECTED: return KERNEL_FULLY_CONNECTED;
        case tflite::BuiltinOperator_GATHER: return KERNEL_GATHER;
        case tflite::BuiltinOperator_GATHER_ND: return KERNEL_GATHER_ND;
        case tflite::BuiltinOperator_GREATER: return KERNEL_GREATER;
        case tflite::BuiltinOperator_GREATER_EQUAL: return KERNEL_GREATER_EQUAL;
        case tflite::BuiltinOperator_HARD_SWISH: return KERNEL_HARD_SWISH;
        case tflite::BuiltinOperator_L2_NORMALIZATION: return KERNEL_L2_NORMALIZATION;
        case tflite::BuiltinOperator_L2_POOL_2D: return KERNEL_L2_POOL_2D;
        case tflite::BuiltinOperator_LEAKY_RELU: return KERNEL_LEAKY_RELU;
        case tflite::BuiltinOperator_LESS: return KERNEL_LESS;
        case tflite::BuiltinOperator_LESS_EQUAL: return KERNEL_LESS_EQUAL;
        case tflite::BuiltinOperator_LOG: return KERNEL_LOG;
        case tflite::BuiltinOperator_LOG_SOFTMAX: return KERNEL_LOG_SOFTMAX;
        case tflite::BuiltinOperator_LOGICAL_AND: return KERNEL_LOGICAL_AND;
        case tflite::BuiltinOperator_LOGICAL_NOT: return KERNEL_LOGICAL_NOT;
        case tflite::BuiltinOperator_LOGICAL_OR: return KERNEL_LOGICAL_OR;
        case tflite::BuiltinOperator_LOGISTIC: return KERNEL_LOGISTIC;
        case tflite::BuiltinOperator_MAX_POOL_2D: return KERNEL_MAX_POOL_2D;
        case tflite::BuiltinOperator_MAXIMUM: return KERNEL_MAXIMUM;
        case tflite::BuiltinOperator_MEAN: return KERNEL_MEAN;
        case tflite::BuiltinOperator_MINIMUM: return KERNEL_MINIMUM;
        case tflite::BuiltinOperator_MIRROR_PAD: return KERNEL_MIRROR_PAD;
        case tflite::BuiltinOperator_MUL: return KERNEL_MUL;
        case tflite::BuiltinOperator_NEG: return KERNEL_NEG;
        case tflite::BuiltinOperator_NOT_EQUAL: return KERNEL_NOT_EQUAL;
        case tflite::BuiltinOperator_PACK: return KERNEL_PACK;
        case tflite::BuiltinOperator_PAD: return KERNEL_PAD;
        case tflite::BuiltinOperator_PADV2: return KERNEL_PADV2;
        case tflite::BuiltinOperator_PRELU: return KERNEL_PRELU;
        case tflite::BuiltinOperator_QUANTIZE: return KERNEL_QUANTIZE;
        case tflite::BuiltinOperator_REDUCE_MAX: return KERNEL_REDUCE_MAX;
        case tflite::BuiltinOperator_REDUCE_MIN: return KERNEL_REDUCE_MIN;
        case tflite::BuiltinOperator_RELU: return KERNEL_RELU;
        case tflite::BuiltinOperator_RELU6: return KERNEL_RELU6;
        case tflite::BuiltinOperator_RESHAPE: return KERNEL_RESHAPE;
        case tflite::BuiltinOperator_RESIZE_BILINEAR: return KERNEL_RESIZE_BILINEAR;
        case tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR: return KERNEL_RESIZE_NEAREST_NEIGHBOR;
        case tflite::BuiltinOperator_REVERSE_V2: return KERNEL_REVERSE_V2;
        case tflite::BuiltinOperator_ROUND: return KERNEL_ROUND;
        case tflite::BuiltinOperator_RSQRT: return KERNEL_RSQRT;
        case tflite::BuiltinOperator_SELECT_V2: return KERNEL_SELECT_V2;
        case tflite::BuiltinOperator_SHAPE: return KERNEL_SHAPE;
        case tflite::BuiltinOperator_SIN: return KERNEL_SIN;
        case tflite::BuiltinOperator_SLICE: return KERNEL_SLICE;
        case tflite::BuiltinOperator_SOFTMAX: return KERNEL_SOFTMAX;
        case tflite::BuiltinOperator_SPACE_TO_BATCH_ND: return KERNEL_SPACE_TO_BATCH_ND;
        case tflite::BuiltinOperator_SPACE_TO_DEPTH: return KERNEL_SPACE_TO_DEPTH;
        case tflite::BuiltinOperator_SPLIT: return KERNEL_SPLIT;
        case tflite::BuiltinOperator_SPLIT_V: return KERNEL_SPLIT_V;
        case tflite::BuiltinOperator_SQRT: return KERNEL_SQRT;
        case tflite::BuiltinOperator_SQUARE: return KERNEL_SQUARE;
        case tflite::BuiltinOperator_SQUARED_DIFFERENCE: return KERNEL_SQUARED_DIFFERENCE;
        case tflite::BuiltinOperator_SQUEEZE: return KERNEL_SQUEEZE;
        case tflite::BuiltinOperator_STRIDED_SLICE: return KERNEL_STRIDED_SLICE;
        case tflite::BuiltinOperator_SUB: return KERNEL_SUB;
        case tflite::BuiltinOperator_SUM: return KERNEL_SUM;
        case tflite::BuiltinOperator_SVDF: return KERNEL_SVDF;
        case tflite::BuiltinOperator_TANH: return KERNEL_TANH;
        case tflite::BuiltinOperator_TRANSPOSE: return KERNEL_TRANSPOSE;
        case tflite::BuiltinOperator_TRANSPOSE_CONV: return KERNEL_TRANSPOSE_CONV;
        case tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM: return KERNEL_UNIDIRECTIONAL_SEQUENCE_LSTM;
        case tflite::BuiltinOperator_UNPACK: return KERNEL_UNPACK;
        case tflite::BuiltinOperator_ZEROS_LIKE: return KERNEL_ZEROS_LIKE;
        
        default:
            MicroPrintf("Operação não mapeada: %d", builtin_op);
            return KERNEL_CONV_2D; // Fallback
    }
}

// Função para analisar modelo e descobrir kernels necessários
size_t AnalyzeModelKernels(const uint8_t* model_data, uint8_t* required_kernels, size_t max_kernels) {
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Erro: Schema do modelo incompatível!");
        return 0;
    }

    const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
    const auto* opcodes = model->operator_codes();
    
    // Array para evitar duplicatas
    bool found_kernels[100] = {false}; // Assumindo max 100 tipos de kernel
    size_t kernel_count = 0;

    MicroPrintf("Analisando kernels necessários no modelo...");

    // Percorre todas as operações no modelo
    for (size_t i = 0; i < subgraph->operators()->size(); ++i) {
        const tflite::Operator* op = subgraph->operators()->Get(i);
        const uint32_t opcode_index = op->opcode_index();
        const tflite::OperatorCode* opcode = opcodes->Get(opcode_index);
        const tflite::BuiltinOperator builtin_code = opcode->builtin_code();
        
        KernelType kernel_type = MapBuiltinOperatorToKernelType(builtin_code);
        
        // Evita duplicatas
        if (!found_kernels[kernel_type] && kernel_count < max_kernels) {
        required_kernels[kernel_count] = static_cast<uint8_t>(kernel_type); // ✅ Converte para uint8_t
        found_kernels[kernel_type] = true;
        kernel_count++;
            MicroPrintf("  + Kernel encontrado: %s", tflite::EnumNameBuiltinOperator(builtin_code));
        }
    }

    MicroPrintf("Total de kernels únicos necessários: %zu", kernel_count);
    return kernel_count;
}

// Função melhorada de inicialização automática
void* InitializeInterpreterAuto(const uint8_t* model_data, uint8_t* tensor_arena, size_t tensor_arena_size) {
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema version mismatch!");
        return nullptr;
    }

    // Primeiro, analisa o modelo para descobrir kernels necessários
    uint8_t required_kernels[50];
    size_t num_kernels = AnalyzeModelKernels(model_data, required_kernels, 50);
    
    if (num_kernels == 0) {
        MicroPrintf("Erro: Nenhum kernel válido encontrado no modelo!");
        return nullptr;
    }

    // Agora inicializa com os kernels descobertos
    TFLM_Instance* instance = new TFLM_Instance();
    instance->resolver = new MutableResolver();

    // Registra apenas os kernels necessários
    MicroPrintf("Registrando %zu kernels descobertos automaticamente:", num_kernels);
    for (size_t i = 0; i < num_kernels; ++i) {
        KernelType kernel_type = static_cast<KernelType>(required_kernels[i]);
        RegisterOp(instance->resolver, kernel_type);
    }
    
    instance->interpreter = new tflite::MicroInterpreter(model, *(instance->resolver), tensor_arena, tensor_arena_size);

    if (instance->interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("Failed to allocate tensors!");
        delete instance->interpreter;
        delete instance->resolver;
        delete instance;
        return nullptr;
    }

    MicroPrintf("Interpreter initialized automatically with %zu kernels.", num_kernels);
    return reinterpret_cast<void*>(instance);
}

// Função de benchmark otimizada
int RunBenchmarkOptimized(const uint8_t* model_data, uint8_t* tensor_arena, size_t tensor_arena_size, int num_invocations) {
    MicroPrintf("--- Iniciando Benchmark Otimizado ---");

    // Usar inicialização automática
    void* interpreter_handle = InitializeInterpreterAuto(model_data, tensor_arena, tensor_arena_size);
    if (!interpreter_handle) {
        MicroPrintf("Erro: Falha na inicialização automática do interpretador.");
        return -1;
    }

    MicroPrintf("Executando %d invocações de benchmark...", num_invocations);

    // Executa benchmark
    for (int i = 0; i < num_invocations; ++i) {
        InvokeInterpreter(interpreter_handle);
        if (i % 10 == 0) { // Log a cada 10 invocações
            MicroPrintf("  - Progresso: %d/%d", i + 1, num_invocations);
        }
    }

    // Cleanup
    DestroyInterpreter(interpreter_handle);
    MicroPrintf("Benchmark otimizado concluído com sucesso!");
    return 0;
}

} // extern "C"