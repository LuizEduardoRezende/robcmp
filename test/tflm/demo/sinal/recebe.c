/*
 * Duas funções C que recebem um código -1 e o devolvem como int32, como uma
 * função do wrapper que recebe um código int8 da RL.
 *
 *   recebe_int8   parâmetro int8_t: como seria um tflm_set_raw_value ingênuo
 *   recebe_int32  parâmetro int32_t: como o tflm_set_raw_value do wrapper é
 *
 * Num parâmetro int8_t, o clang não estende o sinal: confia que quem chamou já
 * estendeu o valor para o registrador inteiro (e no ARM a convenção obriga quem
 * chama a fazer isso). O robcmp não estende, então o -1 (0xFF) é lido como 255.
 * O gcc estende de novo dentro da função, e por isso o erro só aparece com o
 * clang. Com int32 não há o que estender.
 */
#include <stdint.h>
#include <stdio.h>

int32_t recebe_int8(int8_t v) {
	return v;
}

int32_t recebe_int32(int32_t v) {
	return v;
}

static void mostra(const char *chamada, int32_t x) {
	printf("    %-18s devolveu %-4d %s\n", chamada, x, x == -1 ? "correto" : "ERRADO");
}

void mostra_int8(int32_t x) {
	mostra("recebe_int8(-1)", x);
}

void mostra_int32(int32_t x) {
	mostra("recebe_int32(-1)", x);
}
