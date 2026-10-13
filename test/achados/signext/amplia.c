/*
 * Função C que recebe um int8 e o devolve como int32.
 *
 * Compilada pelo clang, ela NÃO estende o sinal do argumento: o clang declara o
 * parâmetro como `i8 signext` e confia que quem chamou já estendeu o valor para
 * o registrador inteiro (no x86_64 e, por obrigação da convenção, no ARM). O
 * corpo vira um único `mov %edi,%eax` -- ou `bx lr` no Cortex-M.
 */
#include <stdint.h>
#include <stdio.h>

int32_t amplia(int8_t v) {
	return v;
}

void mostra(int32_t x) {
	printf("  amplia(-1) devolveu %d\n", x);
}
