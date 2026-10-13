# Demonstrações dos achados -- cada alvo reproduz um problema encontrado durante
# a pesquisa e mostra o resultado na tela. Ver README.md.
#
#   make signext     a RL passa -1 (int8) para uma função C e recebe 255
#   make use         o modo padrão do `use` perde funções e variáveis globais
#   make arena       arena que cada modelo precisa x a que o TCC reservava
#   make             as três
#
# Rodar a partir desta pasta (test/achados). Requer o robcmp compilado
# (build/robcmp) e, para `arena`, a libtensorflow-microlite.a -- gerada pelo
# `make -C test/tflm check`.

ROOT = ../..
ROBCMP = $(ROOT)/build/robcmp
OUT = out

TFLM = $(ROOT)/third-party/tflite-micro
DOWNLOADS = $(TFLM)/tensorflow/lite/micro/tools/make/downloads
GEN = $(TFLM)/gen/linux_x86_64_default_gcc
TFLM_LIB = $(GEN)/lib/libtensorflow-microlite.a
MODELS = $(sort $(wildcard $(ROOT)/test/tflm/models/*.tflite))

all: signext use arena

$(OUT):
	mkdir -p $(OUT)

# ---------------------------------------------------------------------------
# signext: falta de extensão de sinal na chamada RL -> C
# ---------------------------------------------------------------------------

$(OUT)/amplia_clang.o: signext/amplia.c | $(OUT)
	clang -O2 -c $< -o $@

$(OUT)/amplia_gcc.o: signext/amplia.c | $(OUT)
	gcc -O2 -c $< -o $@

$(OUT)/chama.o: signext/chama.rob $(ROBCMP) | $(OUT)
	$(ROBCMP) -o $@ $<

signext: $(OUT)/chama.o $(OUT)/amplia_clang.o $(OUT)/amplia_gcc.o
	@echo "== 1. Como cada lado declara o parametro int8 (LLVM IR)"
	@printf "  RL (robcmp):  "; $(ROBCMP) signext/chama.rob 2>/dev/null | grep '^declare.*@amplia'
	@printf "  C  (clang):   "; clang -O2 -S -emit-llvm -o - signext/amplia.c | grep '^define.*@amplia' | sed 's/ {$$//'
	@echo
	@echo "== 2. O que a funcao C faz com o argumento (codigo de maquina)"
	@echo "  clang:"; objdump -d --no-show-raw-insn $(OUT)/amplia_clang.o | sed -n '/<amplia>:/,/ret/p' | tail -n +2 | sed 's/^/    /'
	@echo "  gcc:";   objdump -d --no-show-raw-insn $(OUT)/amplia_gcc.o   | sed -n '/<amplia>:/,/ret/p' | tail -n +2 | sed 's/^/    /'
	@echo
	@echo "== 3. Rodando: a RL passa -1 e espera -1 de volta"
	@cc -no-pie $(OUT)/chama.o $(OUT)/amplia_clang.o -o $(OUT)/chama_clang
	@cc -no-pie $(OUT)/chama.o $(OUT)/amplia_gcc.o -o $(OUT)/chama_gcc
	@echo "  funcao C compilada pelo clang:"; ./$(OUT)/chama_clang; st=$$?; [ $$st -eq 0 ] && echo "  -> correto" || echo "  -> ERRADO (codigo de saida $$st)"
	@echo "  funcao C compilada pelo gcc:";   ./$(OUT)/chama_gcc;   st=$$?; [ $$st -eq 0 ] && echo "  -> correto" || echo "  -> ERRADO (codigo de saida $$st)"
	@echo

# ---------------------------------------------------------------------------
# use: o modo padrão (só declarações) x -bdep (gramática completa)
# ---------------------------------------------------------------------------

USE_CASES = literal literal-controle globais

use: | $(OUT)
	@echo "== O mesmo programa, com e sem -bdep"
	@for c in $(USE_CASES); do \
		printf "  %-18s" "$$c"; \
		( cd use/$$c && ../../$(ROBCMP) -I . -o ../../$(OUT)/use_$$c.o main.rob > ../../$(OUT)/use_$$c.log 2>&1 ) \
			&& printf "sem -bdep: compila  " || printf "sem -bdep: FALHA    "; \
		( cd use/$$c && ../../$(ROBCMP) -I . -bdep -o ../../$(OUT)/use_$$c.o main.rob > /dev/null 2>&1 ) \
			&& echo "com -bdep: compila" || echo "com -bdep: FALHA"; \
	done
	@echo
	@echo "== Mensagens do modo padrao (sem -bdep)"
	@for c in $(USE_CASES); do \
		if [ -s $(OUT)/use_$$c.log ]; then echo "  $$c:"; sed 's/\x1b\[[0-9;]*m//g; s/^/    /' $(OUT)/use_$$c.log; fi; \
	done
	@echo

# ---------------------------------------------------------------------------
# arena: usada x mínima viável x a do TCC
# ---------------------------------------------------------------------------

ARENA_FLAGS = -std=c++17 -O2 -DTF_LITE_STATIC_MEMORY -fno-rtti -fno-exceptions
ARENA_INCLUDES = -I$(TFLM) -I$(GEN)/genfiles -I$(DOWNLOADS)/flatbuffers/include -I$(DOWNLOADS)/gemmlowp

$(OUT)/arena: arena/arena.cpp | $(OUT)
	@[ -f $(TFLM_LIB) ] || { echo "falta $(TFLM_LIB): rode 'make -C test/tflm check' antes"; exit 1; }
	g++ $(ARENA_FLAGS) $(ARENA_INCLUDES) $< $(TFLM_LIB) -o $@

arena: $(OUT)/arena
	@echo "== Arena por modelo, em bytes (TFLM puro, x86_64)"
	@./$(OUT)/arena $(MODELS)
	@echo

clean:
	rm -rf $(OUT)

.PHONY: all signext use arena clean
