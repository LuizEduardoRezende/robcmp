/*
 * arena -- quanto de arena cada modelo precisa de verdade.
 *
 * Para cada modelo, usando o TFLM puro (sem o wrapper):
 *
 *   usada          arena_used_bytes() depois do AllocateTensors, com uma arena
 *                  folgada. É o número que o TFLM informa.
 *   mínima viável  a menor arena com a qual o AllocateTensors passa, por busca
 *                  binária. Cada tentativa roda num processo filho (fork): com
 *                  arena pequena demais o TFLM pode derrubar o processo em vez
 *                  de devolver erro (RESULTADOS.md, 5.4.12).
 *                  Medida duas vezes: com o buffer começando num endereço
 *                  múltiplo de 16, e começando 8 bytes depois. O TFLM descarta
 *                  os bytes até o próximo múltiplo de 16, então o mesmo modelo
 *                  precisa de mais arena quando o buffer está desalinhado
 *                  (RESULTADOS.md, 5.4.10).
 *   TCC            a arena que os testes do TCC reservavam.
 *
 * O .tflite não guarda o tamanho da arena em lugar nenhum: os dois primeiros
 * números só existem rodando o TFLM sobre o modelo.
 *
 * Uso: ./arena models/hello_world_int8.tflite [...]
 */

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <new>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

// Os kernels dos 7 modelos de test/tflm/models (lista do tflite_info).
static void registra(tflite::MicroMutableOpResolver<32> &r) {
	r.AddAdd(); r.AddAveragePool2D(); r.AddCast(); r.AddConcatenation();
	r.AddConv2D(); r.AddDepthwiseConv2D(); r.AddDiv(); r.AddFullyConnected();
	r.AddGather(); r.AddMaximum(); r.AddMean(); r.AddMinimum(); r.AddMul();
	r.AddReshape(); r.AddSoftmax(); r.AddStridedSlice();
	r.AddEnergy(); r.AddFftAutoScale(); r.AddFilterBank(); r.AddFilterBankLog();
	r.AddFilterBankSpectralSubtraction(); r.AddFilterBankSquareRoot();
	r.AddPCAN(); r.AddRfft(); r.AddWindow();
}

// Arenas usadas nos testes do TCC (0 = o TCC não tinha esse modelo).
static size_t arena_tcc(const std::string &caminho) {
	if (caminho.find("hello_world") != std::string::npos) return 3000;
	if (caminho.find("spam") != std::string::npos) return 4096;
	if (caminho.find("person_detect") != std::string::npos) return 90000;
	if (caminho.find("micro_speech") != std::string::npos) return 12288;
	return 0;
}

// Tenta alocar os tensores com `tamanho` bytes de arena, começando
// `deslocamento` bytes depois de um endereço múltiplo de 16. Devolve os bytes
// usados, ou 0 se falhou.
static size_t tenta(const uint8_t *modelo, size_t tamanho, size_t deslocamento) {
	uint8_t *base = static_cast<uint8_t *>(
		aligned_alloc(16, (tamanho + deslocamento + 15) & ~size_t(15)));
	uint8_t *arena = base + deslocamento;
	tflite::MicroMutableOpResolver<32> resolver;
	registra(resolver);
	alignas(tflite::MicroInterpreter) uint8_t storage[sizeof(tflite::MicroInterpreter)];
	auto *interp = new (storage) tflite::MicroInterpreter(
		tflite::GetModel(modelo), resolver, arena, tamanho);
	size_t usada = interp->AllocateTensors() == kTfLiteOk ? interp->arena_used_bytes() : 0;
	interp->~MicroInterpreter();
	free(base);
	return usada;
}

// A mesma tentativa, isolada num processo filho. true = alocou.
static bool tenta_isolado(const uint8_t *modelo, size_t tamanho, size_t deslocamento) {
	fflush(stdout);		// senão o filho herda e reimprime o que está no buffer
	fflush(stderr);
	pid_t pid = fork();
	if (pid == 0) {
		FILE *e = freopen("/dev/null", "w", stderr);	// o TFLM imprime cada falha
		FILE *o = freopen("/dev/null", "w", stdout);
		(void)e; (void)o;
		_exit(tenta(modelo, tamanho, deslocamento) ? 0 : 1);
	}
	int st = 0;
	waitpid(pid, &st, 0);
	return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}

// Busca binária da menor arena que aloca: `lo` falha, `hi` passa.
static size_t minima_viavel(const uint8_t *modelo, size_t folgada, size_t deslocamento) {
	size_t lo = 0, hi = folgada;
	while (hi - lo > 1) {
		size_t meio = lo + (hi - lo) / 2;
		if (tenta_isolado(modelo, meio, deslocamento))
			hi = meio;
		else
			lo = meio;
	}
	return hi;
}

static bool mede(const char *caminho) {
	std::ifstream f(caminho, std::ios::binary);
	if (!f) {
		fprintf(stderr, "erro: não foi possível abrir %s\n", caminho);
		return false;
	}
	std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	uint8_t *modelo = static_cast<uint8_t *>(aligned_alloc(16, (bytes.size() + 15) & ~size_t(15)));
	memcpy(modelo, bytes.data(), bytes.size());

	const size_t folgada = 512 * 1024;
	size_t usada = tenta(modelo, folgada, 0);
	if (!usada) {
		fprintf(stderr, "erro: %s não aloca nem com %zu bytes\n", caminho, folgada);
		free(modelo);
		return false;
	}

	size_t minima = minima_viavel(modelo, folgada, 0);
	size_t minima8 = minima_viavel(modelo, folgada, 8);

	const char *nome = strrchr(caminho, '/') ? strrchr(caminho, '/') + 1 : caminho;
	size_t tcc = arena_tcc(caminho);
	printf("  %-32s %7zu %10zu %10zu", nome, usada, minima, minima8);
	if (tcc)
		printf(" %8zu %7.0f%%\n", tcc, 100.0 * (tcc - minima) / minima);
	else
		printf(" %8s %8s\n", "-", "-");
	free(modelo);
	return true;
}

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "uso: %s modelo.tflite [...]\n", argv[0]);
		return 2;
	}
	printf("  %-32s %7s %10s %10s %8s %8s\n", "", "", "mínima", "mínima", "", "sobra");
	printf("  %-32s %7s %10s %10s %8s %8s\n", "modelo", "usada", "alinhada", "+8 bytes", "TCC", "no TCC");
	bool ok = true;
	for (int i = 1; i < argc; i++)
		ok &= mede(argv[i]);
	return ok ? 0 : 1;
}
