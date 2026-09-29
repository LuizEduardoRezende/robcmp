/*
 * tflite_info -- inspeciona um arquivo .tflite sem executá-lo.
 *
 * Executável fino sobre o módulo de análise do compilador
 * (src/tinyml/TfliteModel.cpp): o que ele imprime é exatamente o que o robcmp
 * enxerga do modelo em tempo de compilação -- tensores de entrada e saída
 * (tipo, shape, scale, zero_point) e kernels exigidos, cada um com a função do
 * wrapper que o registra.
 *
 * Uso:  ./tflite_info modelo.tflite [outro.tflite ...]
 *       ./tflite_info --kernel-table     tabela kernel -> função do wrapper
 */

#include "tinyml/TfliteModel.h"

#include <cstdio>
#include <cstring>

static void PrintTensor(const TfliteTensor &t, const char *role, unsigned position) {
	printf("  %-6s[%u] tipo=%-8s shape=[", role, position, tfliteTypeName(t.type));
	for (size_t i = 0; i < t.shape.size(); i++)
		printf("%s%d", i ? "," : "", t.shape[i]);
	printf("] elementos=%llu", (unsigned long long)t.elements);

	if (t.quantized) {
		printf("  scale=%g zero_point=%lld", t.scale, (long long)t.zeroPoint);
		if (t.quantChannels > 1)
			printf("  (quantizacao por canal: %u escalas)", t.quantChannels);
	}
	printf("\n");
}

static bool Inspect(const char *path) {
	TfliteModel model;
	std::string error;
	if (!TfliteModel::load(path, model, error)) {
		fprintf(stderr, "erro: %s\n", error.c_str());
		return false;
	}

	printf("=== %s (%zu bytes) ===\n", path, model.bytes.size());
	printf("  versao do schema=%u  subgrafos=%u\n", model.schemaVersion, model.subgraphs);
	printf("  entradas=%zu  saidas=%zu\n", model.inputs.size(), model.outputs.size());

	for (unsigned i = 0; i < model.inputs.size(); i++)
		PrintTensor(model.inputs[i], "input", i);
	for (unsigned i = 0; i < model.outputs.size(); i++)
		PrintTensor(model.outputs[i], "output", i);

	printf("  kernels (%zu):\n", model.kernels.size());
	for (const TfliteKernel &k : model.kernels)
		printf("    %-7s %-38s %s\n", k.custom ? "custom" : "builtin", k.op.c_str(),
		       k.wrapperFunction.empty() ? "SEM SUPORTE" : k.wrapperFunction.c_str());
	printf("\n");
	return true;
}

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "uso: %s modelo.tflite [outro.tflite ...]\n"
		                "     %s --kernel-table\n", argv[0], argv[0]);
		return 2;
	}

	if (strcmp(argv[1], "--kernel-table") == 0) {
		for (const TfliteKernel &k : TfliteModel::supportedKernels())
			printf("%s %s %s\n", k.custom ? "custom" : "builtin", k.op.c_str(),
			       k.wrapperFunction.c_str());
		return 0;
	}

	bool ok = true;
	for (int i = 1; i < argc; i++)
		ok &= Inspect(argv[i]);

	return ok ? 0 : 1;
}
