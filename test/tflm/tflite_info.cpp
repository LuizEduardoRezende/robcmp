/*
 * tflite_info -- inspeciona um arquivo .tflite sem executá-lo.
 *
 * Lê o modelo via FlatBuffers (acesso zero-copy: o arquivo é a estrutura de
 * dados, não há passo de desserialização) e extrai o que a análise estática
 * da seção 5.2 precisa conhecer em tempo de compilação:
 *
 *   - tensores de entrada e saída: tipo, shape, scale e zero_point
 *   - conjunto de kernels exigidos pelo modelo
 *
 * Nenhum interpretador é instanciado e nenhuma arena é alocada. Este é o
 * esqueleto do módulo de elaboração prévia que o compilador vai executar
 * antes de percorrer a AST.
 *
 * Uso:  ./tflite_info modelo.tflite [outro.tflite ...]
 */

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <vector>

static const char *TypeName(tflite::TensorType t) {
	switch (t) {
		case tflite::TensorType_FLOAT32: return "float32";
		case tflite::TensorType_FLOAT16: return "float16";
		case tflite::TensorType_INT32:   return "int32";
		case tflite::TensorType_UINT8:   return "uint8";
		case tflite::TensorType_INT64:   return "int64";
		case tflite::TensorType_STRING:  return "string";
		case tflite::TensorType_BOOL:    return "bool";
		case tflite::TensorType_INT16:   return "int16";
		case tflite::TensorType_INT8:    return "int8";
		case tflite::TensorType_UINT16:  return "uint16";
		case tflite::TensorType_UINT32:  return "uint32";
		default:                         return "outro";
	}
}

// Imprime um tensor de interface (entrada ou saída) do subgrafo.
static void PrintTensor(const tflite::SubGraph *sg, int index,
                        const char *role, int position) {
	const tflite::Tensor *t = sg->tensors()->Get(index);

	printf("  %-6s[%d] tipo=%-8s shape=[", role, position, TypeName(t->type()));

	long long elements = 1;
	for (unsigned i = 0; i < t->shape()->size(); i++) {
		printf("%s%d", i ? "," : "", t->shape()->Get(i));
		elements *= t->shape()->Get(i);
	}
	printf("] elementos=%lld", elements);

	// Parâmetros de quantização assimétrica linear. O schema define a
	// conversão como f = scale * (q - zero_point); é a mesma fórmula que o
	// wrapper aplica hoje em runtime, e que pode ser resolvida em compilação.
	const tflite::QuantizationParameters *q = t->quantization();
	if (q && q->scale() && q->scale()->size() > 0) {
		printf("  scale=%g zero_point=%lld", q->scale()->Get(0),
		       (long long)q->zero_point()->Get(0));
		if (q->scale()->size() > 1)
			printf("  (quantizacao por canal: %u escalas)", q->scale()->size());
	}
	printf("\n");
}

static bool Inspect(const char *path) {
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		fprintf(stderr, "erro: nao foi possivel abrir %s\n", path);
		return false;
	}
	std::vector<char> buffer((std::istreambuf_iterator<char>(file)),
	                          std::istreambuf_iterator<char>());

	// "TFL3" nos bytes 4..7 identifica o formato. Custa nada verificar e evita
	// interpretar lixo como modelo -- a primeira checagem que o compilador
	// deve fazer ao receber um caminho no .rob do usuario.
	if (buffer.size() < 8 || !tflite::ModelBufferHasIdentifier(buffer.data())) {
		fprintf(stderr, "erro: %s nao e um .tflite valido (identificador TFL3 ausente)\n", path);
		return false;
	}

	const tflite::Model *model = tflite::GetModel(buffer.data());
	const tflite::SubGraph *sg = model->subgraphs()->Get(0);

	printf("=== %s (%zu bytes) ===\n", path, buffer.size());
	printf("  versao do schema=%u  subgrafos=%u  tensores=%u  operadores=%u\n",
	       model->version(), model->subgraphs()->size(), sg->tensors()->size(),
	       sg->operators() ? sg->operators()->size() : 0);
	printf("  entradas=%u  saidas=%u\n", sg->inputs()->size(), sg->outputs()->size());

	for (unsigned i = 0; i < sg->inputs()->size(); i++)
		PrintTensor(sg, sg->inputs()->Get(i), "input", i);
	for (unsigned i = 0; i < sg->outputs()->size(); i++)
		PrintTensor(sg, sg->outputs()->Get(i), "output", i);

	// Kernels exigidos. ATENCAO: ler oc->builtin_code() diretamente devolve
	// resultado errado para modelos gerados por conversores antigos, que
	// deixam esse campo zerado e guardam o codigo real em
	// deprecated_builtin_code. GetBuiltinCode() trata os dois casos. A falha e
	// silenciosa: o modelo parece usar ADD quando na verdade usa outra coisa.
	std::set<std::string> builtins, customs;
	for (unsigned i = 0; i < model->operator_codes()->size(); i++) {
		const tflite::OperatorCode *oc = model->operator_codes()->Get(i);
		if (oc->custom_code())
			customs.insert(oc->custom_code()->str());
		else
			builtins.insert(tflite::EnumNameBuiltinOperator(tflite::GetBuiltinCode(oc)));
	}

	printf("  kernels builtin (%zu):", builtins.size());
	for (const std::string &k : builtins)
		printf(" %s", k.c_str());
	printf("\n");

	// Custom ops nao aparecem no enum BuiltinOperator: sao registrados por
	// string, via AddCustom(). O enum KernelType do wrapper atual e cego para
	// eles, e por isso nao consegue rodar o audio preprocessor.
	if (!customs.empty()) {
		printf("  kernels custom  (%zu):", customs.size());
		for (const std::string &k : customs)
			printf(" %s", k.c_str());
		printf("\n");
	}
	printf("\n");
	return true;
}

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "uso: %s modelo.tflite [outro.tflite ...]\n", argv[0]);
		return 2;
	}

	bool ok = true;
	for (int i = 1; i < argc; i++)
		ok &= Inspect(argv[i]);

	return ok ? 0 : 1;
}
