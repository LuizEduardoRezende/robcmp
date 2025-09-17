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
    tflite::RecordingMicroInterpreter* interpreter;
    tflite::RecordingMicroAllocator* allocator;
    MutableResolver* resolver;
};

void RegisterOp(tflite::MicroMutableOpResolver<100>* resolver, KernelType kernel_type) {
    switch (kernel_type) {
        case KERNEL_ABS:
            resolver->AddAbs();
            break;
        case KERNEL_ADD:
            resolver->AddAdd();
            break;
        case KERNEL_ADD_N:
            resolver->AddAddN();
            break;
        case KERNEL_ARG_MAX:
            resolver->AddArgMax();
            break;
        case KERNEL_ARG_MIN:
            resolver->AddArgMin();
            break;
        case KERNEL_ASSIGN_VARIABLE:
            resolver->AddAssignVariable();
            break;
        case KERNEL_AVERAGE_POOL_2D:
            resolver->AddAveragePool2D();
            break;
        case KERNEL_BATCH_MATMUL:
            resolver->AddBatchMatMul();
            break;
        case KERNEL_BATCH_TO_SPACE_ND:
            resolver->AddBatchToSpaceNd();
            break;
        case KERNEL_BROADCAST_ARGS:
            resolver->AddBroadcastArgs();
            break;
        case KERNEL_BROADCAST_TO:
            resolver->AddBroadcastTo();
            break;
        case KERNEL_CALL_ONCE:
            resolver->AddCallOnce();
            break;
        case KERNEL_CAST:
            resolver->AddCast();
            break;
        case KERNEL_CEIL:
            resolver->AddCeil();
            break;
        case KERNEL_CIRCULAR_BUFFER:
            resolver->AddCircularBuffer();
            break;
        case KERNEL_CONCATENATION:
            resolver->AddConcatenation();
            break;
        case KERNEL_CONV_2D:
            resolver->AddConv2D();
            break;
        case KERNEL_COS:
            resolver->AddCos();
            break;
        case KERNEL_CUMSUM:
            resolver->AddCumSum();
            break;
        case KERNEL_DEPTH_TO_SPACE:
            resolver->AddDepthToSpace();
            break;
        case KERNEL_DEPTHWISE_CONV_2D:
            resolver->AddDepthwiseConv2D();
            break;
        case KERNEL_DEQUANTIZE:
            resolver->AddDequantize();
            break;
        case KERNEL_DIV:
            resolver->AddDiv();
            break;
        case KERNEL_ELU:
            resolver->AddElu();
            break;
        case KERNEL_EMBEDDING_LOOKUP:
            resolver->AddEmbeddingLookup();
            break;
        case KERNEL_EQUAL:
            resolver->AddEqual();
            break;
        case KERNEL_ETHOSU:
            resolver->AddEthosU();
            break;
        case KERNEL_EXP:
            resolver->AddExp();
            break;
        case KERNEL_EXPAND_DIMS:
            resolver->AddExpandDims();
            break;
        case KERNEL_FILL:
            resolver->AddFill();
            break;
        case KERNEL_FLOOR:
            resolver->AddFloor();
            break;
        case KERNEL_FLOOR_DIV:
            resolver->AddFloorDiv();
            break;
        case KERNEL_FLOOR_MOD:
            resolver->AddFloorMod();
            break;
        case KERNEL_FULLY_CONNECTED:
            resolver->AddFullyConnected();
            break;
        case KERNEL_GATHER:
            resolver->AddGather();
            break;
        case KERNEL_GATHER_ND:
            resolver->AddGatherNd();
            break;
        case KERNEL_GREATER:
            resolver->AddGreater();
            break;
        case KERNEL_GREATER_EQUAL:
            resolver->AddGreaterEqual();
            break;
        case KERNEL_HARD_SWISH:
            resolver->AddHardSwish();
            break;
        case KERNEL_IF:
            resolver->AddIf();
            break;
        case KERNEL_L2_NORMALIZATION:
            resolver->AddL2Normalization();
            break;
        case KERNEL_L2_POOL_2D:
            resolver->AddL2Pool2D();
            break;
        case KERNEL_LEAKY_RELU:
            resolver->AddLeakyRelu();
            break;
        case KERNEL_LESS:
            resolver->AddLess();
            break;
        case KERNEL_LESS_EQUAL:
            resolver->AddLessEqual();
            break;
        case KERNEL_LOG:
            resolver->AddLog();
            break;
        case KERNEL_LOG_SOFTMAX:
            resolver->AddLogSoftmax();
            break;
        case KERNEL_LOGICAL_AND:
            resolver->AddLogicalAnd();
            break;
        case KERNEL_LOGICAL_NOT:
            resolver->AddLogicalNot();
            break;
        case KERNEL_LOGICAL_OR:
            resolver->AddLogicalOr();
            break;
        case KERNEL_LOGISTIC:
            resolver->AddLogistic();
            break;
        case KERNEL_MAX_POOL_2D:
            resolver->AddMaxPool2D();
            break;
        case KERNEL_MAXIMUM:
            resolver->AddMaximum();
            break;
        case KERNEL_MEAN:
            resolver->AddMean();
            break;
        case KERNEL_MINIMUM:
            resolver->AddMinimum();
            break;
        case KERNEL_MIRROR_PAD:
            resolver->AddMirrorPad();
            break;
        case KERNEL_MUL:
            resolver->AddMul();
            break;
        case KERNEL_NEG:
            resolver->AddNeg();
            break;
        case KERNEL_NOT_EQUAL:
            resolver->AddNotEqual();
            break;
        case KERNEL_PACK:
            resolver->AddPack();
            break;
        case KERNEL_PAD:
            resolver->AddPad();
            break;
        case KERNEL_PADV2:
            resolver->AddPadV2();
            break;
        case KERNEL_PRELU:
            resolver->AddPrelu();
            break;
        case KERNEL_QUANTIZE:
            resolver->AddQuantize();
            break;
        case KERNEL_READ_VARIABLE:
            resolver->AddReadVariable();
            break;
        case KERNEL_REDUCE_MAX:
            resolver->AddReduceMax();
            break;
        case KERNEL_REDUCE_MIN:
            resolver->AddReduceMin();
            break;
        case KERNEL_RELU:
            resolver->AddRelu();
            break;
        case KERNEL_RELU6:
            resolver->AddRelu6();
            break;
        case KERNEL_RESHAPE:
            resolver->AddReshape();
            break;
        case KERNEL_RESIZE_BILINEAR:
            resolver->AddResizeBilinear();
            break;
        case KERNEL_RESIZE_NEAREST_NEIGHBOR:
            resolver->AddResizeNearestNeighbor();
            break;
        case KERNEL_REVERSE_V2:
            resolver->AddReverseV2();
            break;
        case KERNEL_ROUND:
            resolver->AddRound();
            break;
        case KERNEL_RSQRT:
            resolver->AddRsqrt();
            break;
        case KERNEL_SELECT_V2:
            resolver->AddSelectV2();
            break;
        case KERNEL_SHAPE:
            resolver->AddShape();
            break;
        case KERNEL_SIN:
            resolver->AddSin();
            break;
        case KERNEL_SLICE:
            resolver->AddSlice();
            break;
        case KERNEL_SOFTMAX:
            resolver->AddSoftmax();
            break;
        case KERNEL_SPACE_TO_BATCH_ND:
            resolver->AddSpaceToBatchNd();
            break;
        case KERNEL_SPACE_TO_DEPTH:
            resolver->AddSpaceToDepth();
            break;
        case KERNEL_SPLIT:
            resolver->AddSplit();
            break;
        case KERNEL_SPLIT_V:
            resolver->AddSplitV();
            break;
        case KERNEL_SQRT:
            resolver->AddSqrt();
            break;
        case KERNEL_SQUARE:
            resolver->AddSquare();
            break;
        case KERNEL_SQUARED_DIFFERENCE:
            resolver->AddSquaredDifference();
            break;
        case KERNEL_SQUEEZE:
            resolver->AddSqueeze();
            break;
        case KERNEL_STRIDED_SLICE:
            resolver->AddStridedSlice();
            break;
        case KERNEL_SUB:
            resolver->AddSub();
            break;
        case KERNEL_SUM:
            resolver->AddSum();
            break;
        case KERNEL_SVDF:
            resolver->AddSvdf();
            break;
        case KERNEL_TANH:
            resolver->AddTanh();
            break;
        case KERNEL_TRANSPOSE:
            resolver->AddTranspose();
            break;
        case KERNEL_TRANSPOSE_CONV:
            resolver->AddTransposeConv();
            break;
        case KERNEL_UNIDIRECTIONAL_SEQUENCE_LSTM:
            resolver->AddUnidirectionalSequenceLSTM();
            break;
        case KERNEL_UNPACK:
            resolver->AddUnpack();
            break;
        case KERNEL_VAR_HANDLE:
            resolver->AddVarHandle();
            break;
        case KERNEL_WHILE:
            resolver->AddWhile();
            break;
        case KERNEL_ZEROS_LIKE:
            resolver->AddZerosLike();
            break;
        default:
            MicroPrintf("AVISO: Tipo de kernel não mapeado: %d", kernel_type);
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
            MicroPrintf("OPERAÇÃO NÃO MAPEADA: %d (%s)", builtin_op, tflite::EnumNameBuiltinOperator(builtin_op));
            return KERNEL_CONV_2D; // Fallback
    }
}

uintptr_t InitializeInterpreter(const uint8_t* model_data, uint8_t* tensor_arena, const uint8_t* required_kernels, int, int tensor_arena_size, int8_t num_kernels) {
    MicroPrintf("=== InitializeInterpreter ===");
    
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
        MicroPrintf("  Kernel[%d]: %d -> %s", i, required_kernels[i], 
                   (kernel_type == KERNEL_FULLY_CONNECTED) ? "FULLY_CONNECTED" : "OUTRO");
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

    MicroPrintf("=== VERIFICAÇÃO PRE-ALOCAÇÃO ===");
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
    MicroPrintf("=== TENSOR DEBUG ===");
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
    
    MicroPrintf("=== INICIANDO INFERÊNCIA ===");
    TfLiteStatus status = instance->interpreter->Invoke();
    
    if (status != kTfLiteOk) {
        MicroPrintf("ERRO na execução do interpreter: %d", status);
    } else {
        MicroPrintf("✓ Inferência executada com sucesso!");
    }
}

// Função de diagnóstico detalhado
void DiagnoseModel(const uint8_t* model_data, int) {
    MicroPrintf("=== DIAGNÓSTICO COMPLETO DO MODELO ===");
    
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
    MicroPrintf("=== VERIFICAÇÃO DE INTEGRIDADE DO MODELO ===");
    
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

} // extern "C"