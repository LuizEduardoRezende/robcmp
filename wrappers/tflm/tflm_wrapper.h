#pragma once
/*
 * tflm_wrapper -- interface C entre a Robotics Language e o TensorFlow Lite Micro.
 *
 * Todas as funcoes tem ligacao C e sao declaradas, do lado da linguagem, em
 * lib/ai/tflm.rob. As duas listas precisam estar em acordo: nomes, tipos e
 * ordem dos parametros.
 *
 * CONVENCOES DE ASSINATURA
 *
 *   - Handles, indices e contagens sao uint64_t, o tamanho do uint64 da RL em
 *     qualquer alvo. Nunca uintptr_t nem size_t: no Cortex-M eles tem 32 bits,
 *     e um argumento de 64 bits passado a um parametro de 32 desalinha todos os
 *     argumentos seguintes (ver RESULTADOS.md).
 *
 *   - Para cada argumento VETOR, a RL injeta um int32_t com o numero de
 *     elementos, apendado ao fim, na ordem dos vetores. Esses parametros
 *     (values_len, model_size, arena_size) NAO aparecem na declaracao .rob.
 *
 *   - Retorno int8_t: 0 = sucesso, -1 = falha.
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Ciclo de vida do modelo, em tres fases:
 *
 *     h = tflm_create_resolver();
 *     tflm_add_fully_connected(h);        // uma chamada por kernel do modelo
 *     m = tflm_create_model(h, model_data, tensor_arena);
 *
 * O handle do resolver e o do modelo sao o mesmo valor.
 * -------------------------------------------------------------------------*/

// Reserva uma instancia. Devolve o handle do resolver, ou 0 se nao houver slot.
uint64_t tflm_create_resolver(void);

// Constroi o interpretador e aloca os tensores na arena. Devolve o handle do
// modelo, ou 0 em caso de erro (arena insuficiente, schema incompativel,
// argumento nulo).
uint64_t tflm_create_model(uint64_t resolver_handle,
                           const uint8_t *model_data, uint8_t *tensor_arena,
                           int32_t model_size, int32_t arena_size);

// Libera a instancia para reuso. Nao devolve memoria: nao ha heap.
void tflm_free_model(uint64_t handle);

// Executa a inferencia.
int8_t tflm_invoke(uint64_t model_handle);

/* ---------------------------------------------------------------------------
 * Registro de kernels -- uma funcao por kernel (secao 5.1).
 *
 * Cada uma e um simbolo independente: com -ffunction-sections e
 * --gc-sections, so as chamadas entram no binario.
 * -------------------------------------------------------------------------*/

// Operadores builtin
int8_t tflm_add_add(uint64_t resolver_handle);
int8_t tflm_add_average_pool_2d(uint64_t resolver_handle);
int8_t tflm_add_concatenation(uint64_t resolver_handle);
int8_t tflm_add_conv_2d(uint64_t resolver_handle);
int8_t tflm_add_depthwise_conv_2d(uint64_t resolver_handle);
int8_t tflm_add_depth_to_space(uint64_t resolver_handle);
int8_t tflm_add_dequantize(uint64_t resolver_handle);
int8_t tflm_add_embedding_lookup(uint64_t resolver_handle);
int8_t tflm_add_floor(uint64_t resolver_handle);
int8_t tflm_add_fully_connected(uint64_t resolver_handle);
int8_t tflm_add_l2_normalization(uint64_t resolver_handle);
int8_t tflm_add_l2_pool_2d(uint64_t resolver_handle);
int8_t tflm_add_logistic(uint64_t resolver_handle);
int8_t tflm_add_max_pool_2d(uint64_t resolver_handle);
int8_t tflm_add_mul(uint64_t resolver_handle);
int8_t tflm_add_relu(uint64_t resolver_handle);
int8_t tflm_add_relu6(uint64_t resolver_handle);
int8_t tflm_add_reshape(uint64_t resolver_handle);
int8_t tflm_add_resize_bilinear(uint64_t resolver_handle);
int8_t tflm_add_softmax(uint64_t resolver_handle);
int8_t tflm_add_space_to_depth(uint64_t resolver_handle);
int8_t tflm_add_svdf(uint64_t resolver_handle);
int8_t tflm_add_tanh(uint64_t resolver_handle);
int8_t tflm_add_pad(uint64_t resolver_handle);
int8_t tflm_add_gather(uint64_t resolver_handle);
int8_t tflm_add_batch_to_space_nd(uint64_t resolver_handle);
int8_t tflm_add_space_to_batch_nd(uint64_t resolver_handle);
int8_t tflm_add_transpose(uint64_t resolver_handle);
int8_t tflm_add_mean(uint64_t resolver_handle);
int8_t tflm_add_sub(uint64_t resolver_handle);
int8_t tflm_add_div(uint64_t resolver_handle);
int8_t tflm_add_squeeze(uint64_t resolver_handle);
int8_t tflm_add_unidirectional_sequence_lstm(uint64_t resolver_handle);
int8_t tflm_add_strided_slice(uint64_t resolver_handle);
int8_t tflm_add_exp(uint64_t resolver_handle);
int8_t tflm_add_split(uint64_t resolver_handle);
int8_t tflm_add_log_softmax(uint64_t resolver_handle);
int8_t tflm_add_cast(uint64_t resolver_handle);
int8_t tflm_add_prelu(uint64_t resolver_handle);
int8_t tflm_add_maximum(uint64_t resolver_handle);
int8_t tflm_add_arg_max(uint64_t resolver_handle);
int8_t tflm_add_minimum(uint64_t resolver_handle);
int8_t tflm_add_less(uint64_t resolver_handle);
int8_t tflm_add_neg(uint64_t resolver_handle);
int8_t tflm_add_padv2(uint64_t resolver_handle);
int8_t tflm_add_greater(uint64_t resolver_handle);
int8_t tflm_add_greater_equal(uint64_t resolver_handle);
int8_t tflm_add_less_equal(uint64_t resolver_handle);
int8_t tflm_add_slice(uint64_t resolver_handle);
int8_t tflm_add_sin(uint64_t resolver_handle);
int8_t tflm_add_transpose_conv(uint64_t resolver_handle);
int8_t tflm_add_expand_dims(uint64_t resolver_handle);
int8_t tflm_add_equal(uint64_t resolver_handle);
int8_t tflm_add_not_equal(uint64_t resolver_handle);
int8_t tflm_add_log(uint64_t resolver_handle);
int8_t tflm_add_sum(uint64_t resolver_handle);
int8_t tflm_add_sqrt(uint64_t resolver_handle);
int8_t tflm_add_rsqrt(uint64_t resolver_handle);
int8_t tflm_add_shape(uint64_t resolver_handle);
int8_t tflm_add_arg_min(uint64_t resolver_handle);
int8_t tflm_add_reduce_max(uint64_t resolver_handle);
int8_t tflm_add_pack(uint64_t resolver_handle);
int8_t tflm_add_logical_or(uint64_t resolver_handle);
int8_t tflm_add_logical_and(uint64_t resolver_handle);
int8_t tflm_add_logical_not(uint64_t resolver_handle);
int8_t tflm_add_unpack(uint64_t resolver_handle);
int8_t tflm_add_reduce_min(uint64_t resolver_handle);
int8_t tflm_add_floor_div(uint64_t resolver_handle);
int8_t tflm_add_square(uint64_t resolver_handle);
int8_t tflm_add_zeros_like(uint64_t resolver_handle);
int8_t tflm_add_fill(uint64_t resolver_handle);
int8_t tflm_add_floor_mod(uint64_t resolver_handle);
int8_t tflm_add_resize_nearest_neighbor(uint64_t resolver_handle);
int8_t tflm_add_leaky_relu(uint64_t resolver_handle);
int8_t tflm_add_squared_difference(uint64_t resolver_handle);
int8_t tflm_add_mirror_pad(uint64_t resolver_handle);
int8_t tflm_add_abs(uint64_t resolver_handle);
int8_t tflm_add_split_v(uint64_t resolver_handle);
int8_t tflm_add_ceil(uint64_t resolver_handle);
int8_t tflm_add_reverse_v2(uint64_t resolver_handle);
int8_t tflm_add_add_n(uint64_t resolver_handle);
int8_t tflm_add_gather_nd(uint64_t resolver_handle);
int8_t tflm_add_cos(uint64_t resolver_handle);
int8_t tflm_add_elu(uint64_t resolver_handle);
int8_t tflm_add_quantize(uint64_t resolver_handle);
int8_t tflm_add_round(uint64_t resolver_handle);
int8_t tflm_add_hard_swish(uint64_t resolver_handle);
int8_t tflm_add_if(uint64_t resolver_handle);
int8_t tflm_add_while(uint64_t resolver_handle);
int8_t tflm_add_select_v2(uint64_t resolver_handle);
int8_t tflm_add_batch_matmul(uint64_t resolver_handle);
int8_t tflm_add_cumsum(uint64_t resolver_handle);
int8_t tflm_add_call_once(uint64_t resolver_handle);
int8_t tflm_add_broadcast_to(uint64_t resolver_handle);
int8_t tflm_add_var_handle(uint64_t resolver_handle);
int8_t tflm_add_read_variable(uint64_t resolver_handle);
int8_t tflm_add_assign_variable(uint64_t resolver_handle);
int8_t tflm_add_broadcast_args(uint64_t resolver_handle);
int8_t tflm_add_dynamic_update_slice(uint64_t resolver_handle);
int8_t tflm_add_reduce_all(uint64_t resolver_handle);

// Processamento de sinal -- custom ops do pipeline de audio do Micro Speech
int8_t tflm_add_delay(uint64_t resolver_handle);
int8_t tflm_add_energy(uint64_t resolver_handle);
int8_t tflm_add_fft_auto_scale(uint64_t resolver_handle);
int8_t tflm_add_filter_bank(uint64_t resolver_handle);
int8_t tflm_add_filter_bank_log(uint64_t resolver_handle);
int8_t tflm_add_filter_bank_spectral_subtraction(uint64_t resolver_handle);
int8_t tflm_add_filter_bank_square_root(uint64_t resolver_handle);
int8_t tflm_add_framer(uint64_t resolver_handle);
int8_t tflm_add_irfft(uint64_t resolver_handle);
int8_t tflm_add_overlap_add(uint64_t resolver_handle);
int8_t tflm_add_pcan(uint64_t resolver_handle);
int8_t tflm_add_rfft(uint64_t resolver_handle);
int8_t tflm_add_stacker(uint64_t resolver_handle);
int8_t tflm_add_window(uint64_t resolver_handle);

// Outros custom ops
int8_t tflm_add_basic_classifier(uint64_t resolver_handle);
int8_t tflm_add_circular_buffer(uint64_t resolver_handle);
int8_t tflm_add_decode(uint64_t resolver_handle);
int8_t tflm_add_detection_postprocess(uint64_t resolver_handle);
int8_t tflm_add_ethos_u(uint64_t resolver_handle);

/* ---------------------------------------------------------------------------
 * Acesso aos tensores -- espelha a sintaxe nativa.
 * -------------------------------------------------------------------------*/

// m.input0, m.output0, ... Devolvem 0 se o indice nao existir no modelo.
uint64_t tflm_input(uint64_t model_handle, uint64_t index);
uint64_t tflm_output(uint64_t model_handle, uint64_t index);

uint64_t tflm_input_count(uint64_t model_handle);
uint64_t tflm_output_count(uint64_t model_handle);

// Numero de elementos do tensor (m.input0.size).
uint64_t tflm_tensor_size(uint64_t tensor_handle);

// Valores reais, com conversao automatica:
//   q = round(v/scale) + zero_point      v = scale * (q - zero_point)
// Um elemento. Indice fora do tensor: escrita ignorada, leitura devolve 0.
void tflm_set_value(uint64_t tensor_handle, uint64_t index, float value);
float tflm_get_value(uint64_t tensor_handle, uint64_t index);

// Valores reais, o tensor inteiro. values_len precisa ser exatamente o numero
// de elementos do tensor; senao devolve -1 sem copiar nada.
int8_t tflm_set_values(uint64_t tensor_handle, const float *values, int32_t values_len);
int8_t tflm_get_values(uint64_t tensor_handle, float *values, int32_t values_len);

// Codigos crus, sem conversao (m.input0.raw). Tipo do vetor igual ao do
// tensor e values_len igual ao numero de elementos; senao -1 sem copiar nada.
int8_t tflm_set_raw_int8(uint64_t tensor_handle, const int8_t *values, int32_t values_len);
int8_t tflm_get_raw_int8(uint64_t tensor_handle, int8_t *values, int32_t values_len);
int8_t tflm_set_raw_int16(uint64_t tensor_handle, const int16_t *values, int32_t values_len);
int8_t tflm_get_raw_int16(uint64_t tensor_handle, int16_t *values, int32_t values_len);

// Codigo cru de um elemento (m.input0.raw[k]), para tensor int8 ou int16. O
// valor trafega como int32 nos dois sentidos: um int8/int16 escalar dependeria
// de quem estende o sinal, e o robcmp nao marca signext (RESULTADOS.md, 5.2.7).
// Indice fora do tensor ou tensor de outro tipo: escrita ignorada, leitura
// devolve 0. Valor fora da faixa do tipo: saturado.
void tflm_set_raw_value(uint64_t tensor_handle, uint64_t index, int32_t value);
int32_t tflm_get_raw_value(uint64_t tensor_handle, uint64_t index);

// Utilitario: imprime um float pelo log do TFLM (stderr no x86, UART no MCU).
void tflm_print_float(float value);

#ifdef __cplusplus
}
#endif
