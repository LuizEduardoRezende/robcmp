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

/* Ciclo de vida do modelo, em tres fases.
 *
 * Uso em .rob (modo ai.tflm) ou emitido pelo compilador (sintaxe nativa):
 *
 *     h = tflm_create_resolver();
 *     tflm_add_fully_connected(h);   // uma chamada por kernel do modelo
 *     m = tflm_create_model(h, model_data, tensor_arena);
 *
 * Os parametros int32_t finais sao os tamanhos que a RL injeta
 * automaticamente, um por argumento vetor, na ordem em que os vetores
 * aparecem. Nao devem ser escritos na declaracao .rob.
 */

// Reserva uma instancia e devolve o handle do resolver. 0 = sem slot livre.
uintptr_t tflm_create_resolver(void);

// Constroi o interpretador e aloca os tensores. Devolve o handle do modelo
// (mesmo valor do handle do resolver) ou 0 em caso de erro.
uintptr_t tflm_create_model(uintptr_t resolver_handle,
                            const uint8_t* model_data, uint8_t* tensor_arena,
                            int32_t model_size, int32_t arena_size);

// Libera a instancia. Nao devolve memoria (nao ha heap): apenas marca o slot
// como reutilizavel e destroi o interpretador.
void tflm_free_model(uintptr_t handle);

// Obtém um tensor de entrada do interpretador
// Retorna 0 em caso de erro, senão retorna handle do tensor
uintptr_t GetInputTensor(uintptr_t instance_handle, size_t index);

// Obtém um tensor de saída do interpretador
// Retorna 0 em caso de erro, senão retorna handle do tensor
uintptr_t GetOutputTensor(uintptr_t instance_handle, size_t index);

// Executa uma inferência no interpretador
// Executa a inferencia. Devolve 0 em caso de sucesso, -1 em caso de falha.
// O status ja era verificado internamente e descartado num log; agora chega
// a quem chamou.
int8_t InvokeInterpreter(uintptr_t instance_handle);

// Define o valor de um elemento específico em um tensor de entrada
// Converte automaticamente o valor float para o tipo correto do tensor
void SetTensorValue(uintptr_t tensor_handle, size_t index, float value);

// Obtém o valor de um elemento específico de um tensor de saída como float
// Converte automaticamente do tipo do tensor para float
float GetTensorAsFloat(uintptr_t tensor_handle, size_t index);

// Retorna o número total de elementos em um tensor
size_t GetTensorSize(uintptr_t tensor_handle);


/* ---------------------------------------------------------------------------
 * Registro de kernels -- uma funcao por kernel (secao 5.1).
 *
 * O compilador emite uma chamada para cada kernel que a analise estatica do
 * .tflite identificou como necessario. Kernels nao chamados sao descartados na
 * linkedicao por --gc-sections, sem configuracao manual.
 *
 * Parametro: handle do resolver, obtido na criacao do modelo.
 * Retorno:   0 = registrado, -1 = falha.
 * -------------------------------------------------------------------------*/

int8_t tflm_add_add(uintptr_t resolver_handle);
int8_t tflm_add_average_pool_2d(uintptr_t resolver_handle);
int8_t tflm_add_concatenation(uintptr_t resolver_handle);
int8_t tflm_add_conv_2d(uintptr_t resolver_handle);
int8_t tflm_add_depthwise_conv_2d(uintptr_t resolver_handle);
int8_t tflm_add_depth_to_space(uintptr_t resolver_handle);
int8_t tflm_add_dequantize(uintptr_t resolver_handle);
int8_t tflm_add_embedding_lookup(uintptr_t resolver_handle);
int8_t tflm_add_floor(uintptr_t resolver_handle);
int8_t tflm_add_fully_connected(uintptr_t resolver_handle);
int8_t tflm_add_l2_normalization(uintptr_t resolver_handle);
int8_t tflm_add_l2_pool_2d(uintptr_t resolver_handle);
int8_t tflm_add_logistic(uintptr_t resolver_handle);
int8_t tflm_add_max_pool_2d(uintptr_t resolver_handle);
int8_t tflm_add_mul(uintptr_t resolver_handle);
int8_t tflm_add_relu(uintptr_t resolver_handle);
int8_t tflm_add_relu6(uintptr_t resolver_handle);
int8_t tflm_add_reshape(uintptr_t resolver_handle);
int8_t tflm_add_resize_bilinear(uintptr_t resolver_handle);
int8_t tflm_add_softmax(uintptr_t resolver_handle);
int8_t tflm_add_space_to_depth(uintptr_t resolver_handle);
int8_t tflm_add_svdf(uintptr_t resolver_handle);
int8_t tflm_add_tanh(uintptr_t resolver_handle);
int8_t tflm_add_pad(uintptr_t resolver_handle);
int8_t tflm_add_gather(uintptr_t resolver_handle);
int8_t tflm_add_batch_to_space_nd(uintptr_t resolver_handle);
int8_t tflm_add_space_to_batch_nd(uintptr_t resolver_handle);
int8_t tflm_add_transpose(uintptr_t resolver_handle);
int8_t tflm_add_mean(uintptr_t resolver_handle);
int8_t tflm_add_sub(uintptr_t resolver_handle);
int8_t tflm_add_div(uintptr_t resolver_handle);
int8_t tflm_add_squeeze(uintptr_t resolver_handle);
int8_t tflm_add_unidirectional_sequence_lstm(uintptr_t resolver_handle);
int8_t tflm_add_strided_slice(uintptr_t resolver_handle);
int8_t tflm_add_exp(uintptr_t resolver_handle);
int8_t tflm_add_split(uintptr_t resolver_handle);
int8_t tflm_add_log_softmax(uintptr_t resolver_handle);
int8_t tflm_add_cast(uintptr_t resolver_handle);
int8_t tflm_add_prelu(uintptr_t resolver_handle);
int8_t tflm_add_maximum(uintptr_t resolver_handle);
int8_t tflm_add_arg_max(uintptr_t resolver_handle);
int8_t tflm_add_minimum(uintptr_t resolver_handle);
int8_t tflm_add_less(uintptr_t resolver_handle);
int8_t tflm_add_neg(uintptr_t resolver_handle);
int8_t tflm_add_padv2(uintptr_t resolver_handle);
int8_t tflm_add_greater(uintptr_t resolver_handle);
int8_t tflm_add_greater_equal(uintptr_t resolver_handle);
int8_t tflm_add_less_equal(uintptr_t resolver_handle);
int8_t tflm_add_slice(uintptr_t resolver_handle);
int8_t tflm_add_sin(uintptr_t resolver_handle);
int8_t tflm_add_transpose_conv(uintptr_t resolver_handle);
int8_t tflm_add_expand_dims(uintptr_t resolver_handle);
int8_t tflm_add_equal(uintptr_t resolver_handle);
int8_t tflm_add_not_equal(uintptr_t resolver_handle);
int8_t tflm_add_log(uintptr_t resolver_handle);
int8_t tflm_add_sum(uintptr_t resolver_handle);
int8_t tflm_add_sqrt(uintptr_t resolver_handle);
int8_t tflm_add_rsqrt(uintptr_t resolver_handle);
int8_t tflm_add_shape(uintptr_t resolver_handle);
int8_t tflm_add_arg_min(uintptr_t resolver_handle);
int8_t tflm_add_reduce_max(uintptr_t resolver_handle);
int8_t tflm_add_pack(uintptr_t resolver_handle);
int8_t tflm_add_logical_or(uintptr_t resolver_handle);
int8_t tflm_add_logical_and(uintptr_t resolver_handle);
int8_t tflm_add_logical_not(uintptr_t resolver_handle);
int8_t tflm_add_unpack(uintptr_t resolver_handle);
int8_t tflm_add_reduce_min(uintptr_t resolver_handle);
int8_t tflm_add_floor_div(uintptr_t resolver_handle);
int8_t tflm_add_square(uintptr_t resolver_handle);
int8_t tflm_add_zeros_like(uintptr_t resolver_handle);
int8_t tflm_add_fill(uintptr_t resolver_handle);
int8_t tflm_add_floor_mod(uintptr_t resolver_handle);
int8_t tflm_add_resize_nearest_neighbor(uintptr_t resolver_handle);
int8_t tflm_add_leaky_relu(uintptr_t resolver_handle);
int8_t tflm_add_squared_difference(uintptr_t resolver_handle);
int8_t tflm_add_mirror_pad(uintptr_t resolver_handle);
int8_t tflm_add_abs(uintptr_t resolver_handle);
int8_t tflm_add_split_v(uintptr_t resolver_handle);
int8_t tflm_add_ceil(uintptr_t resolver_handle);
int8_t tflm_add_reverse_v2(uintptr_t resolver_handle);
int8_t tflm_add_add_n(uintptr_t resolver_handle);
int8_t tflm_add_gather_nd(uintptr_t resolver_handle);
int8_t tflm_add_cos(uintptr_t resolver_handle);
int8_t tflm_add_elu(uintptr_t resolver_handle);
int8_t tflm_add_quantize(uintptr_t resolver_handle);
int8_t tflm_add_round(uintptr_t resolver_handle);
int8_t tflm_add_hard_swish(uintptr_t resolver_handle);
int8_t tflm_add_if(uintptr_t resolver_handle);
int8_t tflm_add_while(uintptr_t resolver_handle);
int8_t tflm_add_select_v2(uintptr_t resolver_handle);
int8_t tflm_add_batch_matmul(uintptr_t resolver_handle);
int8_t tflm_add_cumsum(uintptr_t resolver_handle);
int8_t tflm_add_call_once(uintptr_t resolver_handle);
int8_t tflm_add_broadcast_to(uintptr_t resolver_handle);
int8_t tflm_add_var_handle(uintptr_t resolver_handle);
int8_t tflm_add_read_variable(uintptr_t resolver_handle);
int8_t tflm_add_assign_variable(uintptr_t resolver_handle);
int8_t tflm_add_broadcast_args(uintptr_t resolver_handle);

// Custom ops (registrados por string) e builtins recentes.
// Os nove Signal* sao o pipeline de audio do Micro Speech.
int8_t tflm_add_basic_classifier(uintptr_t resolver_handle);
int8_t tflm_add_circular_buffer(uintptr_t resolver_handle);
int8_t tflm_add_decode(uintptr_t resolver_handle);
int8_t tflm_add_delay(uintptr_t resolver_handle);
int8_t tflm_add_detection_postprocess(uintptr_t resolver_handle);
int8_t tflm_add_dynamic_update_slice(uintptr_t resolver_handle);
int8_t tflm_add_energy(uintptr_t resolver_handle);
int8_t tflm_add_ethos_u(uintptr_t resolver_handle);
int8_t tflm_add_fft_auto_scale(uintptr_t resolver_handle);
int8_t tflm_add_filter_bank(uintptr_t resolver_handle);
int8_t tflm_add_filter_bank_log(uintptr_t resolver_handle);
int8_t tflm_add_filter_bank_spectral_subtraction(uintptr_t resolver_handle);
int8_t tflm_add_filter_bank_square_root(uintptr_t resolver_handle);
int8_t tflm_add_framer(uintptr_t resolver_handle);
int8_t tflm_add_irfft(uintptr_t resolver_handle);
int8_t tflm_add_overlap_add(uintptr_t resolver_handle);
int8_t tflm_add_pcan(uintptr_t resolver_handle);
int8_t tflm_add_reduce_all(uintptr_t resolver_handle);
int8_t tflm_add_rfft(uintptr_t resolver_handle);
int8_t tflm_add_stacker(uintptr_t resolver_handle);
int8_t tflm_add_window(uintptr_t resolver_handle);

/* Nas tres funcoes abaixo o parametro final values_len NAO deve ser escrito na
 * declaracao .rob: a RL o injeta automaticamente, um por argumento vetor. As
 * demais funcoes do wrapper nao tem argumento vetor e portanto nao o recebem.
 */
// Função para obter um tensor completo como array
void GetTensorArray(uintptr_t tensor_handle, float* values, size_t max_count, int32_t values_len);

// Função para definir um tensor completo a partir de um array
void SetTensorArray(uintptr_t tensor_handle, const float* values, size_t count, int32_t values_len);

// Função para definir um tensor completo a partir de um array de inteiros (cópia bruta)
void SetTensorFromIntArray(uintptr_t tensor_handle, const int16_t* values, size_t count, int32_t values_len);

// Função para obter o número de tensores de entrada
size_t GetInputTensorCount(uintptr_t instance_handle);

// Função para obter o número de tensores de saída
size_t GetOutputTensorCount(uintptr_t instance_handle);

// Função simples para imprimir float com quebras de linha
void PrintFloat(float value);

#ifdef __cplusplus
}
#endif
