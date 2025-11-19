# Makefile para testes unitários TFLM
# Baseado no padrão do test/general mas adaptado para TFLM

ALL_ROB=$(wildcard *.rob)
ALL_TT=$(patsubst %.rob,%.tt,$(ALL_ROB))
ALL_OO=$(patsubst %.rob,out/%.o,$(ALL_ROB))
ALL_EXE=$(patsubst %.rob,out/%,$(ALL_ROB))

# Diretórios
BUILD_DIR=../../build
OUT_DIR=out
WRAPPER_DIR=../../wrappers/tflm
TFLM_DIR=../../third-party/tflite-micro

# Verificar sistema operacional
OS=$(shell uname -o)

# Ferramentas e bibliotecas
ROBCMP=$(BUILD_DIR)/robcmp
CLANG=clang++
WRAPPER_LIB=$(OUT_DIR)/libtflm_wrapper.a
TFLM_LIB=$(TFLM_DIR)/gen/*/lib/libtensorflow-microlite.a

# Adicionar estas novas flags de compilação
WRAPPER_FLAGS = -fno-rtti -fno-exceptions -DTF_LITE_STATIC_MEMORY=1

# Includes para o wrapper
WRAPPER_INCLUDES = -I../../ -I$(WRAPPER_DIR) -I$(TFLM_DIR) 
WRAPPER_INCLUDES += -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
WRAPPER_INCLUDES += -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/gemmlowp
WRAPPER_INCLUDES += -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/gen/*/genfiles
WRAPPER_INCLUDES += -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/ruy

# Flags de linkagem para redução de tamanho
LDFLAGS_OPTIMIZE = -Wl,--gc-sections -Wl,--strip-all -Wl,--strip-debug
LDFLAGS_OPTIMIZE += -Wl,--discard-all -Wl,--discard-locals -flto
LDFLAGS_OPTIMIZE += -Wl,--build-id=none -Wl,--relax
#LDFLAGS_OPTIMIZE += -static -Wl, (aumenta muito o tamanho)
#LDFLAGS_OPTIMIZE += -Wl,--lto-whole-program-visibility (erro de build)
LDFLAGS_OPTIMIZE += -flto=thin

# Target padrão
all: out ${ALL_TT}

# Manter arquivos .o após linking (não remover automaticamente)
.SECONDARY:

# Criar diretório de saída
out:
	mkdir -p out

# Compilar o wrapper TFLM localmente
wrapper: $(WRAPPER_LIB)

# Compilar wrapper para arquivo objeto
$(OUT_DIR)/tflm_wrapper.o: $(WRAPPER_DIR)/tflm_wrapper.cpp $(WRAPPER_DIR)/kernel_config.h | out
	@echo "Compilando TFLM wrapper..."
	$(CLANG) $(WRAPPER_FLAGS) $(WRAPPER_INCLUDES) \
		-std=c++14 -c $(WRAPPER_DIR)/tflm_wrapper.cpp -o $@

# Criar biblioteca estática do wrapper
$(WRAPPER_LIB): $(OUT_DIR)/tflm_wrapper.o
	@echo "Criando biblioteca estática do wrapper..."
	ar rcs $@ $<
	@echo "✓ Wrapper compilado: $@"

# Configurar permissões necessárias
setup:
	@chmod +x runtest.sh 2>/dev/null || true

# Executar todos os testes
test: setup wrapper ${ALL_TT}
	@echo "Todos os testes TFLM executados"

# Pattern rule para executar teste individual
%.tt : %.rob
	@./runtest.sh $< $(patsubst %.rob,%,$<)

# Compilar .rob para .o
out/%.o : %.rob ${ROBCMP} Makefile
	@mkdir -p out
	${ROBCMP} -Oz -I ../../lib $< -o $@


# Linkar .o com bibliotecas TFLM para criar executável
out/% : out/%.o $(WRAPPER_LIB)
	${CLANG} $(LDFLAGS_OPTIMIZE) -o $@ $< ${WRAPPER_LIB} ${TFLM_LIB}

# Limpeza
clean:
	@echo "Limpando arquivos de teste TFLM..."
	@rm -f ${ALL_OO} ${ALL_EXE}
	@rm -rf out

# Listar testes disponíveis
list:
	@echo "Testes TFLM disponíveis:"
	@for file in $(ALL_ROB); do \
		echo "  make $$(basename $$file .rob).tt"; \
	done

# Help
help:
	@echo "Makefile para testes unitários TFLM"
	@echo ""
	@echo "Uso:"
	@echo "  make          # Compila wrapper e todos os executáveis"
	@echo "  make wrapper  # Compila apenas o wrapper TFLM"
	@echo "  make list     # Lista testes disponíveis"
	@echo "  make clean    # Remove arquivos gerados"
	@echo "  make help     # Mostra esta ajuda"
	@echo ""
	@echo "Para compilar teste individual:"
	@echo "  make out/arquivo    # Compila executável out/arquivo a partir de arquivo.rob"
	@echo ""
	@echo "Processo de compilação:"
	@echo "  1. Compila wrapper TFLM localmente (usa kernel_config.h)"
	@echo "  2. Compila arquivos .rob para .o"
	@echo "  3. Linka com bibliotecas TFLM"
	@echo ""
	@echo "Para otimizar tamanho, edite: ../../wrappers/tflm/kernel_config.h"
	@echo ""
	@echo "Arquivos de teste encontrados:"
	@ls -1 *.rob 2>/dev/null || echo "  (nenhum arquivo .rob encontrado)"

.PHONY: all setup test clean list help wrapper