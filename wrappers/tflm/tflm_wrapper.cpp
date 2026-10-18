#include "third-party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "third-party/tflite-micro/tensorflow/lite/schema/schema_generated.h"
#include "third-party/tflite-micro/tensorflow/lite/c/common.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/micro_log.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/micro_utils.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/micro_arena_constants.h"
#include "tflm_wrapper.h"

#include <cstring>
#include <cstdint>
#include <new>

extern "C" {

// Logging de diagnostico. Controlado pela build (-DTFLM_DEBUG_LOGS=1), nao por
// edicao de arquivo. Desligado, TFLM_LOG vira no-op e o compilador descarta os
// argumentos junto.
#ifndef TFLM_DEBUG_LOGS
#define TFLM_DEBUG_LOGS 0
#endif
// Logs de diagnostico usam TFLM_LOG, nunca MicroPrintf diretamente. Antes, o
// no-op era aplicado redefinindo o proprio MicroPrintf, o que desligava junto
// o PrintFloat (hoje tflm_print_float) -- funcao do usuario, nao log de diagnostico.
#if TFLM_DEBUG_LOGS
#define TFLM_LOG(...) MicroPrintf(__VA_ARGS__)
#else
#define TFLM_LOG(...) ((void)0)
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

// Menor arena em que o TFLM consegue ao menos montar suas estruturas de
// controle: SingleArenaBufferAllocator, GreedyMemoryPlanner (o planejador
// padrao) e MicroAllocator, cada um com folga de alinhamento, mais o
// alinhamento inicial da arena a 16 bytes. Abaixo disso o TFLM cai em vez de
// devolver erro. Acima, falta de memoria vira falha controlada no
// AllocateTensors.
static constexpr size_t kArenaMinima =
    sizeof(tflite::SingleArenaBufferAllocator) + alignof(tflite::SingleArenaBufferAllocator) +
    sizeof(tflite::GreedyMemoryPlanner) + alignof(tflite::GreedyMemoryPlanner) +
    sizeof(tflite::MicroAllocator) + alignof(tflite::MicroAllocator) +
    tflite::MicroArenaBufferAlignment();

/* Handles cruzam a fronteira com a RL como uint64_t -- o tamanho do uint64 da
 * linguagem em QUALQUER alvo. Nao usar uintptr_t nem size_t na assinatura: no
 * Cortex-M eles tem 32 bits, enquanto a RL passa 64. Pelo AAPCS um argumento
 * de 64 bits ocupa um par de registradores, e o C leria a metade alta do
 * handle no lugar do argumento seguinte -- tflm_input(h, 1) devolveria sempre
 * o tensor 0, sem erro nenhum. No x86_64 os tipos coincidem por acaso, e por
 * isso o defeito passou despercebido no TCC. */
static void *de_handle(uint64_t h) {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(h));
}

static uint64_t para_handle(const void *p) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
}

static TFLM_Instance *instance_from_handle(uint64_t h) {
    return static_cast<TFLM_Instance*>(de_handle(h));
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

// O handle aponta para a TFLM_Instance, nao para o resolver. Acessar o campo
// explicitamente evita depender de `resolver` ser o primeiro membro da struct.
static MutableResolver *resolver_from_handle(uint64_t h) {
    return h ? &instance_from_handle(h)->resolver : nullptr;
}

int8_t tflm_add_add(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddAdd() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_average_pool_2d(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddAveragePool2D() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_concatenation(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddConcatenation() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_conv_2d(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddConv2D() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_depthwise_conv_2d(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDepthwiseConv2D() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_depth_to_space(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDepthToSpace() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_dequantize(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDequantize() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_embedding_lookup(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddEmbeddingLookup() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_floor(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFloor() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_fully_connected(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFullyConnected() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_l2_normalization(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddL2Normalization() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_l2_pool_2d(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddL2Pool2D() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_logistic(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLogistic() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_max_pool_2d(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddMaxPool2D() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_mul(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddMul() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_relu(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddRelu() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_relu6(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddRelu6() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_reshape(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddReshape() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_resize_bilinear(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddResizeBilinear() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_softmax(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSoftmax() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_space_to_depth(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSpaceToDepth() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_svdf(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSvdf() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_tanh(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddTanh() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_pad(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddPad() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_gather(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddGather() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_batch_to_space_nd(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddBatchToSpaceNd() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_space_to_batch_nd(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSpaceToBatchNd() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_transpose(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddTranspose() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_mean(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddMean() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_sub(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSub() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_div(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDiv() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_squeeze(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSqueeze() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_unidirectional_sequence_lstm(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddUnidirectionalSequenceLSTM() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_strided_slice(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddStridedSlice() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_exp(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddExp() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_split(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSplit() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_log_softmax(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLogSoftmax() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_cast(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddCast() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_prelu(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddPrelu() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_maximum(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddMaximum() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_arg_max(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddArgMax() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_minimum(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddMinimum() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_less(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLess() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_neg(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddNeg() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_padv2(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddPadV2() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_greater(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddGreater() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_greater_equal(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddGreaterEqual() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_less_equal(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLessEqual() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_slice(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSlice() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_sin(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSin() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_transpose_conv(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddTransposeConv() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_expand_dims(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddExpandDims() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_equal(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddEqual() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_not_equal(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddNotEqual() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_log(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLog() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_sum(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSum() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_sqrt(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSqrt() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_rsqrt(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddRsqrt() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_shape(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddShape() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_arg_min(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddArgMin() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_reduce_max(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddReduceMax() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_pack(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddPack() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_logical_or(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLogicalOr() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_logical_and(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLogicalAnd() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_logical_not(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLogicalNot() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_unpack(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddUnpack() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_reduce_min(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddReduceMin() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_floor_div(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFloorDiv() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_square(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSquare() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_zeros_like(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddZerosLike() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_fill(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFill() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_floor_mod(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFloorMod() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_resize_nearest_neighbor(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddResizeNearestNeighbor() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_leaky_relu(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddLeakyRelu() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_squared_difference(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSquaredDifference() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_mirror_pad(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddMirrorPad() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_abs(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddAbs() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_split_v(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSplitV() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_ceil(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddCeil() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_reverse_v2(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddReverseV2() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_add_n(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddAddN() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_gather_nd(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddGatherNd() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_cos(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddCos() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_elu(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddElu() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_quantize(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddQuantize() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_round(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddRound() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_hard_swish(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddHardSwish() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_if(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddIf() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_while(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddWhile() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_select_v2(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddSelectV2() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_batch_matmul(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddBatchMatMul() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_cumsum(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddCumSum() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_call_once(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddCallOnce() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_broadcast_to(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddBroadcastTo() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_var_handle(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddVarHandle() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_read_variable(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddReadVariable() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_assign_variable(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddAssignVariable() == kTfLiteOk) ? 0 : -1;
}

int8_t tflm_add_broadcast_args(uint64_t resolver_handle) {
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
int8_t tflm_add_basic_classifier(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddBasicClassifier() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_circular_buffer(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddCircularBuffer() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_decode(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDecode() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_delay(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDelay() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_detection_postprocess(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDetectionPostprocess() == kTfLiteOk) ? 0 : -1;
}

// builtin
int8_t tflm_add_dynamic_update_slice(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddDynamicUpdateSlice() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_energy(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddEnergy() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_ethos_u(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddEthosU() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_fft_auto_scale(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFftAutoScale() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_filter_bank(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFilterBank() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_filter_bank_log(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFilterBankLog() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_filter_bank_spectral_subtraction(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFilterBankSpectralSubtraction() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_filter_bank_square_root(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFilterBankSquareRoot() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_framer(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddFramer() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_irfft(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddIrfft() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_overlap_add(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddOverlapAdd() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_pcan(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddPCAN() == kTfLiteOk) ? 0 : -1;
}

// builtin
int8_t tflm_add_reduce_all(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddReduceAll() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_rfft(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddRfft() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_stacker(uint64_t resolver_handle) {
    MutableResolver *r = resolver_from_handle(resolver_handle);
    if (!r) return -1;
    return (r->AddStacker() == kTfLiteOk) ? 0 : -1;
}

// custom op
int8_t tflm_add_window(uint64_t resolver_handle) {
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

uint64_t tflm_create_resolver(void) {
    for (int i = 0; i < TFLM_MAX_MODELS; ++i) {
        if (!g_instances[i].in_use) {
            TFLM_Instance *inst = &g_instances[i];
            inst->in_use = true;
            inst->allocator = nullptr;
            inst->interpreter = nullptr;
            new (&inst->resolver) MutableResolver();
            return para_handle(inst);
        }
    }
    TFLM_LOG("ERRO: limite de %d modelos simultaneos atingido", TFLM_MAX_MODELS);
    return 0;
}

uint64_t tflm_create_model(uint64_t resolver_handle,
                            const uint8_t *model_data, uint8_t *tensor_arena,
                            int32_t model_size, int32_t arena_size) {
    (void)model_size;  // o tamanho do modelo vem do proprio FlatBuffer

    if (resolver_handle == 0 || model_data == nullptr || tensor_arena == nullptr) {
        TFLM_LOG("ERRO: argumento nulo em tflm_create_model");
        return 0;
    }

    TFLM_Instance *inst = instance_from_handle(resolver_handle);

    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TFLM_LOG("ERRO: schema incompativel. Versao %d, esperada %d",
                    model->version(), TFLITE_SCHEMA_VERSION);
        return 0;
    }

    // O TFLM coloca tres estruturas de controle dentro da propria arena, em
    // alocacoes seguidas cujo resultado ele NAO verifica (so ha TFLITE_DCHECK,
    // desligado em release): se a arena nao comporta as tres, o MicroAllocator
    // e construido sobre nullptr e o programa cai. Por isso a verificacao
    // "if (!allocator)" logo abaixo nunca disparava -- o crash vem antes.
    // Recusamos a arena aqui, antes de chamar o TFLM.
    if ((size_t)arena_size < kArenaMinima) {
        TFLM_LOG("ERRO: arena de %d bytes abaixo do minimo de %u",
                 arena_size, (unsigned)kArenaMinima);
        return 0;
    }

    // MicroAllocator aloca dentro da propria arena: nao usa heap.
    inst->allocator = tflite::MicroAllocator::Create(tensor_arena, arena_size);
    if (!inst->allocator) {
        TFLM_LOG("ERRO: falha ao criar MicroAllocator");
        return 0;
    }

    inst->interpreter = new (inst->interpreter_storage)
        tflite::MicroInterpreter(model, inst->resolver, inst->allocator, nullptr);

    TfLiteStatus status = inst->interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        TFLM_LOG("ERRO: AllocateTensors falhou (status %d). Arena: %d bytes, usada: %zu",
                    status, arena_size, inst->interpreter->arena_used_bytes());
        inst->interpreter->~MicroInterpreter();
        inst->interpreter = nullptr;
        return 0;
    }

    TFLM_LOG("Modelo pronto. Arena: %d bytes, usada: %zu",
                arena_size, inst->interpreter->arena_used_bytes());
    return resolver_handle;
}

void tflm_free_model(uint64_t handle) {
    if (handle == 0) return;
    TFLM_Instance *inst = instance_from_handle(handle);
    if (inst->interpreter) {
        inst->interpreter->~MicroInterpreter();
        inst->interpreter = nullptr;
    }
    inst->allocator = nullptr;
    inst->in_use = false;
}

/* ---------------------------------------------------------------------------
 * Acesso aos tensores -- espelha a sintaxe nativa (test/tflm/native-syntax.rob),
 * em que cada tensor e um membro do modelo, numerado a partir de 0.
 *
 *   tflm_input(m, 0)            m.input0
 *   tflm_output(m, 0)           m.output0
 *   tflm_set_values(t, v)       m.input0 = v             valores reais
 *   tflm_get_values(t, v)       v = m.output0
 *   tflm_set_value(t, j, x)     m.input0[j] = x
 *   tflm_get_value(t, j)        x = m.output0[j]
 *   tflm_set_raw_int8(t, v)     m.input0.raw = v         codigos crus
 *   tflm_get_raw_int8(t, v)     v = m.output0.raw
 *   tflm_set_raw_value(t, j, c) m.input0.raw[j] = c
 *   tflm_get_raw_value(t, j)    c = m.output0.raw[j]
 *   tflm_tensor_size(t)         m.input0.size
 * -------------------------------------------------------------------------*/

static TfLiteTensor *tensor_de(uint64_t handle) {
    return static_cast<TfLiteTensor*>(de_handle(handle));
}

static uint64_t elementos(const TfLiteTensor *t) {
    return (t && t->dims) ? static_cast<uint64_t>(tflite::ElementCount(*t->dims)) : 0;
}

static tflite::MicroInterpreter *interpretador(uint64_t model_handle) {
    return model_handle ? instance_from_handle(model_handle)->interpreter : nullptr;
}

uint64_t tflm_input(uint64_t model_handle, uint64_t index) {
    tflite::MicroInterpreter *it = interpretador(model_handle);
    if (!it || index >= it->inputs_size()) return 0;
    return para_handle(it->input(static_cast<size_t>(index)));
}

uint64_t tflm_output(uint64_t model_handle, uint64_t index) {
    tflite::MicroInterpreter *it = interpretador(model_handle);
    if (!it || index >= it->outputs_size()) return 0;
    return para_handle(it->output(static_cast<size_t>(index)));
}

uint64_t tflm_input_count(uint64_t model_handle) {
    tflite::MicroInterpreter *it = interpretador(model_handle);
    return it ? it->inputs_size() : 0;
}

uint64_t tflm_output_count(uint64_t model_handle) {
    tflite::MicroInterpreter *it = interpretador(model_handle);
    return it ? it->outputs_size() : 0;
}

uint64_t tflm_tensor_size(uint64_t tensor_handle) {
    return tensor_handle ? elementos(tensor_de(tensor_handle)) : 0;
}

int8_t tflm_invoke(uint64_t model_handle) {
    tflite::MicroInterpreter *it = interpretador(model_handle);
    if (!it) return -1;
    return it->Invoke() == kTfLiteOk ? 0 : -1;
}

/* ---------------------------------------------------------------------------
 * Valores reais -- conversao float <-> quantizado (secao 5.1, item B6).
 *
 * O despacho por tipo fica num ponto so, tensor_write/tensor_read. E ali que
 * mora a verificacao de limites: um indice fora do tensor e recusado. A
 * unificacao do B6 havia perdido essa verificacao, que o wrapper do TCC tinha
 * -- uma escrita com indice grande demais corromperia a arena.
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

static int8_t tensor_write(TfLiteTensor *t, uint64_t index, float value) {
    if (!t || !t->data.data || index >= elementos(t)) return -1;
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
            TFLM_LOG("ERRO: tipo de tensor nao suportado na escrita: %d", t->type);
            return -1;
    }
}

static float tensor_read(const TfLiteTensor *t, uint64_t index) {
    if (!t || !t->data.data || index >= elementos(t)) return 0.0f;
    switch (t->type) {
        case kTfLiteFloat32: return t->data.f[index];
        case kTfLiteInt8:    return from_storage(t, t->data.int8[index]);
        case kTfLiteUInt8:   return from_storage(t, t->data.uint8[index]);
        case kTfLiteInt16:   return from_storage(t, t->data.i16[index]);
        case kTfLiteInt32:   return from_storage(t, t->data.i32[index]);
        case kTfLiteInt64:   return from_storage(t, t->data.i64[index]);
        default:
            TFLM_LOG("ERRO: tipo de tensor nao suportado na leitura: %d", t->type);
            return 0.0f;
    }
}

void tflm_set_value(uint64_t tensor_handle, uint64_t index, float value) {
    if (tensor_handle) tensor_write(tensor_de(tensor_handle), index, value);
}

float tflm_get_value(uint64_t tensor_handle, uint64_t index) {
    return tensor_handle ? tensor_read(tensor_de(tensor_handle), index) : 0.0f;
}

// O tamanho vem so do que a RL injeta (values_len), e precisa ser exatamente o
// numero de elementos do tensor. Nao ha mais um `count` escrito pelo usuario
// ao lado do tamanho injetado -- herança do TCC que permitia os dois
// divergirem e a copia sair do vetor.
int8_t tflm_set_values(uint64_t tensor_handle, const float *values, int32_t values_len) {
    TfLiteTensor *t = tensor_handle ? tensor_de(tensor_handle) : nullptr;
    if (!t || !values || values_len < 0 || (uint64_t)values_len != elementos(t)) return -1;
    for (int32_t i = 0; i < values_len; ++i)
        if (tensor_write(t, (uint64_t)i, values[i]) != 0) return -1;
    return 0;
}

int8_t tflm_get_values(uint64_t tensor_handle, float *values, int32_t values_len) {
    const TfLiteTensor *t = tensor_handle ? tensor_de(tensor_handle) : nullptr;
    if (!t || !values || values_len < 0 || (uint64_t)values_len != elementos(t)) return -1;
    for (int32_t i = 0; i < values_len; ++i)
        values[i] = tensor_read(t, (uint64_t)i);
    return 0;
}

/* ---------------------------------------------------------------------------
 * Codigos crus -- sem conversao float <-> quantizado.
 *
 * Para dados que ja estao na representacao do modelo: saida de um modelo que
 * alimenta outro (o pipeline do micro speech) ou dados de sensor ja em int8.
 * Pela conversao acima eles seriam requantizados -- no micro speech, 39% da
 * entrada do classificador era corrompida.
 *
 * Exigentes de proposito: o tipo do tensor precisa ser o do vetor, e o vetor
 * precisa ter exatamente o numero de elementos do tensor; senao devolvem -1 sem
 * copiar nada. Sao separadas por tipo porque a RL injeta a contagem de
 * ELEMENTOS, nao de bytes.
 * -------------------------------------------------------------------------*/

static TfLiteTensor *tensor_cru(uint64_t handle, TfLiteType tipo, int32_t len) {
    TfLiteTensor *t = handle ? tensor_de(handle) : nullptr;
    if (!t || !t->data.data || t->type != tipo) return nullptr;
    if (len < 0 || (uint64_t)len != elementos(t)) return nullptr;
    return t;
}

int8_t tflm_set_raw_int8(uint64_t tensor_handle, const int8_t *values, int32_t values_len) {
    TfLiteTensor *t = tensor_cru(tensor_handle, kTfLiteInt8, values_len);
    if (!t || !values) return -1;
    memcpy(t->data.int8, values, (size_t)values_len * sizeof(int8_t));
    return 0;
}

int8_t tflm_get_raw_int8(uint64_t tensor_handle, int8_t *values, int32_t values_len) {
    TfLiteTensor *t = tensor_cru(tensor_handle, kTfLiteInt8, values_len);
    if (!t || !values) return -1;
    memcpy(values, t->data.int8, (size_t)values_len * sizeof(int8_t));
    return 0;
}

int8_t tflm_set_raw_int16(uint64_t tensor_handle, const int16_t *values, int32_t values_len) {
    TfLiteTensor *t = tensor_cru(tensor_handle, kTfLiteInt16, values_len);
    if (!t || !values) return -1;
    memcpy(t->data.i16, values, (size_t)values_len * sizeof(int16_t));
    return 0;
}

int8_t tflm_get_raw_int16(uint64_t tensor_handle, int16_t *values, int32_t values_len) {
    TfLiteTensor *t = tensor_cru(tensor_handle, kTfLiteInt16, values_len);
    if (!t || !values) return -1;
    memcpy(values, t->data.i16, (size_t)values_len * sizeof(int16_t));
    return 0;
}

// Um elemento, int8 ou int16. O valor e int32 para nao depender de extensao
// de sinal na chamada (ver o header).
static TfLiteTensor *tensor_cru_elemento(uint64_t handle, uint64_t index) {
    TfLiteTensor *t = handle ? tensor_de(handle) : nullptr;
    if (!t || !t->data.data || index >= elementos(t)) return nullptr;
    if (t->type != kTfLiteInt8 && t->type != kTfLiteInt16) return nullptr;
    return t;
}

void tflm_set_raw_value(uint64_t tensor_handle, uint64_t index, int32_t value) {
    TfLiteTensor *t = tensor_cru_elemento(tensor_handle, index);
    if (!t) return;
    if (t->type == kTfLiteInt8)
        t->data.int8[index] = (int8_t)clamp64(value, -128, 127);
    else
        t->data.i16[index] = (int16_t)clamp64(value, -32768, 32767);
}

int32_t tflm_get_raw_value(uint64_t tensor_handle, uint64_t index) {
    const TfLiteTensor *t = tensor_cru_elemento(tensor_handle, index);
    if (!t) return 0;
    return t->type == kTfLiteInt8 ? t->data.int8[index] : t->data.i16[index];
}

void tflm_print_float(float value) {
    MicroPrintf("%f", value);
}

} // extern "C"