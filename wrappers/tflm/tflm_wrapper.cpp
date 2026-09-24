#include "third-party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "third-party/tflite-micro/tensorflow/lite/schema/schema_generated.h"
#include "third-party/tflite-micro/tensorflow/lite/c/common.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/micro_log.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/micro_utils.h"
#include <cstring>
#include <cstdint>
#include <new>

extern "C" {

// Logging. Controlado pela build (-DTFLM_DEBUG_LOGS=1), nao por edicao de
// arquivo. Desligado, MicroPrintf vira no-op e o compilador descarta os
// argumentos junto.
#ifndef TFLM_DEBUG_LOGS
#define TFLM_DEBUG_LOGS 0
#endif
#if !TFLM_DEBUG_LOGS
#define MicroPrintf(...) ((void)0)
#endif

// Capacidade do resolver. E parametro de template, logo fixado quando o
// wrapper vira .a -- nao pode ser derivado por programa do usuario. Cobre os
// 121 kernels que o TFLM expoe, com folga.
#ifndef TFLM_MAX_OPS
#define TFLM_MAX_OPS 128
#endif

// Numero maximo de modelos simultaneos. Instancias vivem em armazenamento
// estatico: nao ha heap em firmware.
#ifndef TFLM_MAX_MODELS
#define TFLM_MAX_MODELS 4
#endif

using MutableResolver = tflite::MicroMutableOpResolver<TFLM_MAX_OPS>;

struct TFLM_Instance {
    MutableResolver resolver;
    tflite::MicroAllocator *allocator;
    tflite::MicroInterpreter *interpreter;
    // Armazenamento do interpretador: construido com placement new, sem heap.
    alignas(tflite::MicroInterpreter) uint8_t interpreter_storage[sizeof(tflite::MicroInterpreter)];
    bool in_use;
};

static TFLM_Instance g_instances[TFLM_MAX_MODELS];

static TFLM_Instance *instance_from_handle(uintptr_t h) {
    return reinterpret_cast<TFLM_Instance*>(h);
}

/* ---------------------------------------------------------------------------
 * Registro de kernels -- uma funcao por kernel.
 *
 * Cada funcao e um simbolo independente. Compilado com -ffunction-sections e
 * linkado com --gc-sections, um programa que referencia apenas algumas delas
 * descarta as demais, e com elas os Register_* correspondentes do TFLM. E o
 * que substitui a selecao manual por #define do kernel_config.h.
 *
 * Retorno: 0 = registrado, -1 = falha (handle invalido ou resolver cheio).
 * -------------------------------------------------------------------------*/

static MutableResolver *resolver_from_handle(uintptr_t h) {
    return reinterpret_cast<MutableResolver*>(h);
}

int8_t tflm_add_add(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddAdd() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_average_pool_2d(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddAveragePool2D() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_concatenation(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddConcatenation() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_conv_2d(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddConv2D() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_depthwise_conv_2d(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDepthwiseConv2D() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_depth_to_space(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDepthToSpace() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_dequantize(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDequantize() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_embedding_lookup(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddEmbeddingLookup() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_floor(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFloor() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_fully_connected(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFullyConnected() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_l2_normalization(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddL2Normalization() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_l2_pool_2d(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddL2Pool2D() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_logistic(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLogistic() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_max_pool_2d(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddMaxPool2D() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_mul(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddMul() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_relu(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddRelu() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_relu6(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddRelu6() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_reshape(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddReshape() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_resize_bilinear(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddResizeBilinear() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_softmax(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSoftmax() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_space_to_depth(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSpaceToDepth() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_svdf(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSvdf() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_tanh(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddTanh() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_pad(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddPad() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_gather(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddGather() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_batch_to_space_nd(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddBatchToSpaceNd() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_space_to_batch_nd(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSpaceToBatchNd() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_transpose(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddTranspose() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_mean(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddMean() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_sub(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSub() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_div(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDiv() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_squeeze(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSqueeze() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_unidirectional_sequence_lstm(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddUnidirectionalSequenceLSTM() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_strided_slice(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddStridedSlice() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_exp(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddExp() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_split(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSplit() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_log_softmax(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLogSoftmax() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_cast(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddCast() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_prelu(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddPrelu() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_maximum(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddMaximum() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_arg_max(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddArgMax() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_minimum(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddMinimum() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_less(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLess() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_neg(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddNeg() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_padv2(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddPadV2() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_greater(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddGreater() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_greater_equal(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddGreaterEqual() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_less_equal(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLessEqual() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_slice(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSlice() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_sin(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSin() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_transpose_conv(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddTransposeConv() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_expand_dims(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddExpandDims() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_equal(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddEqual() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_not_equal(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddNotEqual() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_log(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLog() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_sum(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSum() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_sqrt(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSqrt() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_rsqrt(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddRsqrt() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_shape(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddShape() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_arg_min(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddArgMin() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_reduce_max(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddReduceMax() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_pack(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddPack() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_logical_or(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLogicalOr() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_logical_and(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLogicalAnd() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_logical_not(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLogicalNot() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_unpack(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddUnpack() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_reduce_min(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddReduceMin() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_floor_div(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFloorDiv() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_square(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSquare() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_zeros_like(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddZerosLike() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_fill(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFill() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_floor_mod(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFloorMod() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_resize_nearest_neighbor(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddResizeNearestNeighbor() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_leaky_relu(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLeakyRelu() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_squared_difference(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSquaredDifference() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_mirror_pad(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddMirrorPad() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_abs(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddAbs() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_split_v(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSplitV() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_ceil(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddCeil() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_reverse_v2(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddReverseV2() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_add_n(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddAddN() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_gather_nd(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddGatherNd() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_cos(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddCos() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_elu(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddElu() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_quantize(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddQuantize() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_round(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddRound() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_hard_swish(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddHardSwish() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_if(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddIf() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_while(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddWhile() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_select_v2(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSelectV2() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_batch_matmul(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddBatchMatMul() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_cumsum(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddCumSum() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_call_once(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddCallOnce() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_broadcast_to(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddBroadcastTo() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_var_handle(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddVarHandle() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_read_variable(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddReadVariable() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_assign_variable(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddAssignVariable() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_broadcast_args(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddBroadcastArgs() == kTfLiteOk) ? 0 : -1;
}



/* ---------------------------------------------------------------------------
 * Kernels que o enum KernelType do TCC nao alcancava.
 *
 * Os 19 primeiros sao custom ops: o TFLM os registra por string, via
 * AddCustom("SignalRfft", ...), e no .tflite eles aparecem em
 * OperatorCode.custom_code, nunca em builtin_code. Um enum derivado de
 * BuiltinOperator e cego para eles por construcao -- por isso faltavam.
 *
 * Os dois ultimos sao builtins que passaram a existir no upstream depois do
 * pin usado no TCC.
 *
 * Os nove Signal* formam o pipeline de audio do Micro Speech, exigidos pelo
 * audio_preprocessor. Sem eles esse modelo nao roda.
 * -------------------------------------------------------------------------*/

// custom op
int8_t tflm_add_basic_classifier(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddBasicClassifier() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_circular_buffer(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddCircularBuffer() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_decode(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDecode() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_delay(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDelay() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_detection_postprocess(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDetectionPostprocess() == kTfLiteOk) ? 0 : -1;
}

// builtin
int8_t tflm_add_dynamic_update_slice(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDynamicUpdateSlice() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_energy(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddEnergy() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_ethos_u(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddEthosU() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_fft_auto_scale(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFftAutoScale() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_filter_bank(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFilterBank() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_filter_bank_log(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFilterBankLog() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_filter_bank_spectral_subtraction(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFilterBankSpectralSubtraction() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_filter_bank_square_root(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFilterBankSquareRoot() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_framer(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFramer() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_irfft(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddIrfft() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_overlap_add(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddOverlapAdd() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_pcan(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddPCAN() == kTfLiteOk) ? 0 : -1;
}

// builtin
int8_t tflm_add_reduce_all(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddReduceAll() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_rfft(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddRfft() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_stacker(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddStacker() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_window(uintptr_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddWindow() == kTfLiteOk) ? 0 : -1;
}

/* ---------------------------------------------------------------------------
 * Ciclo de vida do modelo, em tres fases.
 *
 * O compilador (sintaxe nativa) ou o usuario (modo ai.tflm) emite:
 *
 *     h = tflm_create_resolver();
 *     tflm_add_fully_connected(h);      // uma chamada por kernel do modelo
 *     m = tflm_create_model(h, model_data, tensor_arena);
 *
 * A fase intermediaria e o que permite a eliminacao por linkedicao: so os
 * kernels efetivamente chamados entram no binario.
 *
 * Handles sao ponteiros para instancias em armazenamento estatico. O handle do
 * resolver e o do modelo sao o mesmo valor -- a distincao nos nomes existe
 * para deixar as fases legiveis em quem le o .rob.
 * -------------------------------------------------------------------------*/

uintptr_t tflm_create_resolver(void) {
    for (int i = 0; i < TFLM_MAX_MODELS; ++i) {
        if (!g_instances[i].in_use) {
            TFLM_Instance *inst = &g_instances[i];
            inst->in_use = true;
            inst->allocator = nullptr;
            inst->interpreter = nullptr;
            new (&inst->resolver) MutableResolver();
            return reinterpret_cast<uintptr_t>(inst);
        }
    }
    MicroPrintf("ERRO: limite de %d modelos simultaneos atingido", TFLM_MAX_MODELS);
    return 0;
}

uintptr_t tflm_create_model(uintptr_t resolver_handle,
                            const uint8_t *model_data, uint8_t *tensor_arena,
                            int32_t model_size, int32_t arena_size) {
    (void)model_size;  // o tamanho do modelo vem do proprio FlatBuffer

    if (resolver_handle == 0 || model_data == nullptr || tensor_arena == nullptr) {
        MicroPrintf("ERRO: argumento nulo em tflm_create_model");
        return 0;
    }

    TFLM_Instance *inst = instance_from_handle(resolver_handle);

    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("ERRO: schema incompativel. Versao %d, esperada %d",
                    model->version(), TFLITE_SCHEMA_VERSION);
        return 0;
    }

    // MicroAllocator aloca dentro da propria arena: nao usa heap.
    inst->allocator = tflite::MicroAllocator::Create(tensor_arena, arena_size);
    if (!inst->allocator) {
        MicroPrintf("ERRO: falha ao criar MicroAllocator");
        return 0;
    }

    inst->interpreter = new (inst->interpreter_storage)
        tflite::MicroInterpreter(model, inst->resolver, inst->allocator, nullptr);

    TfLiteStatus status = inst->interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        MicroPrintf("ERRO: AllocateTensors falhou (status %d). Arena: %d bytes, usada: %zu",
                    status, arena_size, inst->interpreter->arena_used_bytes());
        inst->interpreter->~MicroInterpreter();
        inst->interpreter = nullptr;
        return 0;
    }

    MicroPrintf("Modelo pronto. Arena: %d bytes, usada: %zu",
                arena_size, inst->interpreter->arena_used_bytes());
    return resolver_handle;
}

void tflm_free_model(uintptr_t handle) {
    if (handle == 0) return;
    TFLM_Instance *inst = instance_from_handle(handle);
    if (inst->interpreter) {
        inst->interpreter->~MicroInterpreter();
        inst->interpreter = nullptr;
    }
    inst->allocator = nullptr;
    inst->in_use = false;
}

uintptr_t GetInputTensor(uintptr_t instance_handle, size_t index) {
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

uintptr_t GetOutputTensor(uintptr_t instance_handle, size_t index) {
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

/* ---------------------------------------------------------------------------
 * Conversao de tipos -- ponto unico (secao 5.1, item B6).
 *
 * Antes, o despacho por tipo de tensor estava replicado em seis switches, com
 * cada copia tendo seu proprio tratamento de ponteiro nulo, arredondamento e
 * clamping. SetTensorValue sozinho ocupava 222 linhas para escrever um valor.
 *
 * Duas assimetrias que a duplicacao escondia, agora corrigidas: a leitura
 * cobria 4 tipos contra 6 da escrita (faltavam int16 e int64 -- e a entrada do
 * audio_preprocessor e int16), e o clamping so existia em alguns ramos.
 *
 * Quando a analise estatica da secao 5.2 estiver em pe, scale e zero_point
 * passam a ser constantes de compilacao e o compilador pode emitir a conversao
 * ja resolvida. Esta camada continua servindo o modo ai.tflm, que decide em
 * runtime.
 * -------------------------------------------------------------------------*/

// Le scale/zero_point do tensor. Devolve false se nao houver quantizacao
// afim, caso em que a conversao e uma simples truncagem com arredondamento.
static bool tensor_quant(const TfLiteTensor *t, float *scale, int32_t *zero_point) {
    if (t->quantization.type != kTfLiteAffineQuantization) return false;
    const TfLiteAffineQuantization *q =
        reinterpret_cast<const TfLiteAffineQuantization*>(t->quantization.params);
    if (!q || !q->scale || !q->zero_point) return false;
    if (q->scale->size == 0 || q->zero_point->size == 0) return false;
    *scale = q->scale->data[0];
    *zero_point = q->zero_point->data[0];
    return true;
}

static int64_t clamp64(int64_t v, int64_t lo, int64_t hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// float -> representacao do tensor. q = round(v/scale) + zero_point
static int64_t to_storage(const TfLiteTensor *t, float v) {
    float scale; int32_t zp;
    if (tensor_quant(t, &scale, &zp) && scale != 0.0f)
        return (int64_t)roundf(v / scale) + zp;
    return (int64_t)roundf(v);
}

// representacao do tensor -> float. f = scale * (q - zero_point)
static float from_storage(const TfLiteTensor *t, int64_t q) {
    float scale; int32_t zp;
    if (tensor_quant(t, &scale, &zp))
        return scale * (float)(q - zp);
    return (float)q;
}

// Escreve um float na posicao index, convertendo conforme o tipo do tensor.
// Devolve 0 em caso de sucesso, -1 se o tipo nao for suportado ou os dados
// nao estiverem alocados.
static int8_t tensor_write(TfLiteTensor *t, size_t index, float value) {
    if (!t || !t->data.data) return -1;
    switch (t->type) {
        case kTfLiteFloat32:
            t->data.f[index] = value;
            return 0;
        case kTfLiteInt8:
            t->data.int8[index] = (int8_t)clamp64(to_storage(t, value), -128, 127);
            return 0;
        case kTfLiteUInt8:
            t->data.uint8[index] = (uint8_t)clamp64(to_storage(t, value), 0, 255);
            return 0;
        case kTfLiteInt16:
            t->data.i16[index] = (int16_t)clamp64(to_storage(t, value), -32768, 32767);
            return 0;
        case kTfLiteInt32:
            t->data.i32[index] = (int32_t)clamp64(to_storage(t, value), INT32_MIN, INT32_MAX);
            return 0;
        case kTfLiteInt64:
            t->data.i64[index] = to_storage(t, value);
            return 0;
        default:
            MicroPrintf("ERRO: tipo de tensor nao suportado na escrita: %d", t->type);
            return -1;
    }
}

// Le a posicao index como float, desquantizando conforme o tipo.
static float tensor_read(const TfLiteTensor *t, size_t index) {
    if (!t || !t->data.data) return 0.0f;
    switch (t->type) {
        case kTfLiteFloat32: return t->data.f[index];
        case kTfLiteInt8:    return from_storage(t, t->data.int8[index]);
        case kTfLiteUInt8:   return from_storage(t, t->data.uint8[index]);
        case kTfLiteInt16:   return from_storage(t, t->data.i16[index]);
        case kTfLiteInt32:   return from_storage(t, t->data.i32[index]);
        case kTfLiteInt64:   return from_storage(t, t->data.i64[index]);
        default:
            MicroPrintf("ERRO: tipo de tensor nao suportado na leitura: %d", t->type);
            return 0.0f;
    }
}

void SetTensorValue(uintptr_t tensor_handle, size_t index, float value) {
    if (tensor_handle == 0) return;
    TfLiteTensor *t = reinterpret_cast<TfLiteTensor*>(tensor_handle);
    tensor_write(t, index, value);
}

float GetTensorAsFloat(uintptr_t tensor_handle, size_t index) {
    if (tensor_handle == 0) return 0.0f;
    return tensor_read(reinterpret_cast<const TfLiteTensor*>(tensor_handle), index);
}

size_t GetTensorSize(uintptr_t tensor_handle) {
    if (tensor_handle == 0) return 0;
    const TfLiteTensor *t = reinterpret_cast<const TfLiteTensor*>(tensor_handle);
    if (!t->dims) return 0;
    // ElementCount deriva o total das dimensoes. A versao anterior fazia
    // bytes / tamanho_do_elemento, com um switch proprio por tipo -- que
    // errava para tensores cujo bytes nao e multiplo exato do elemento.
    int n = tflite::ElementCount(*t->dims);
    return n > 0 ? (size_t)n : 0;
}

void SetTensorArray(uintptr_t tensor_handle, const float* values, size_t count, int32_t values_len) {
    if (tensor_handle == 0 || !values) return;
    TfLiteTensor *t = reinterpret_cast<TfLiteTensor*>(tensor_handle);
    for (size_t i = 0; i < count; ++i)
        tensor_write(t, i, values[i]);
}

void SetTensorFromIntArray(uintptr_t tensor_handle, const int16_t* values, size_t count, int32_t values_len) {
    if (tensor_handle == 0 || !values) return;
    TfLiteTensor *t = reinterpret_cast<TfLiteTensor*>(tensor_handle);
    for (size_t i = 0; i < count; ++i)
        tensor_write(t, i, (float)values[i]);
}

void GetTensorArray(uintptr_t tensor_handle, float* values, size_t max_count, int32_t values_len) {
    if (tensor_handle == 0 || !values) return;
    const TfLiteTensor *t = reinterpret_cast<const TfLiteTensor*>(tensor_handle);
    for (size_t i = 0; i < max_count; ++i)
        values[i] = tensor_read(t, i);
}

// Função para alocar e retornar um array preenchido com os valores do tensor

size_t GetInputTensorCount(uintptr_t instance_handle) {
    if (instance_handle == 0) return 0;
    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    return instance->interpreter->inputs_size();
}

size_t GetOutputTensorCount(uintptr_t instance_handle) {
    if (instance_handle == 0) return 0;
    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    return instance->interpreter->outputs_size();
}

int8_t InvokeInterpreter(uintptr_t instance_handle) {
    if (instance_handle == 0) {
        MicroPrintf("ERRO: instance_handle nulo");
        return -1;
    }

    TFLM_Instance *instance = instance_from_handle(instance_handle);
    if (instance->interpreter == nullptr) {
        MicroPrintf("ERRO: modelo nao inicializado");
        return -1;
    }

    TfLiteStatus status = instance->interpreter->Invoke();
    if (status != kTfLiteOk) {
        MicroPrintf("ERRO na inferencia: status %d", status);
        return -1;
    }
    return 0;
}

// Função simples para imprimir float com quebras de linha
void PrintFloat(float value) {
    MicroPrintf("\n%.6f\n", value);
}

} // extern "C"