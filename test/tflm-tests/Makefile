# Makefile para testes unitários TFLM
# Baseado no padrão do test/general mas adaptado para TFLM

ALL_ROB=$(wildcard *.rob)
ALL_TT=$(patsubst %.rob,%.tt,$(ALL_ROB))
ALL_OO=$(patsubst %.rob,out/%.o,$(ALL_ROB))
ALL_EXE=$(patsubst %.rob,out/%,$(ALL_ROB))

# Diretórios
BUILD_DIR=../../build
OUT_DIR=out

# Verificar sistema operacional
OS=$(shell uname -o)

# Ferramentas e bibliotecas
ROBCMP=$(BUILD_DIR)/robcmp
CLANG=clang++
WRAPPER_LIB=$(BUILD_DIR)/libtflm_wrapper_static.a
TFLM_LIB=../../third-party/tflite-micro/gen/*/lib/libtensorflow-microlite.a

# Flags de otimização
ifndef OPT
OPT=-Oz
endif

# Target padrão
all: out test

# Manter arquivos .o após linking (não remover automaticamente)
.SECONDARY:

# Criar diretório de saída
out:
	mkdir -p out

# Configurar permissões necessárias
setup:
	@chmod +x runtest.sh 2>/dev/null || true

# Executar todos os testes
test: setup ${ALL_TT}
	@echo "Todos os testes TFLM executados"

# Pattern rule para executar teste individual
%.tt : %.rob
	@./runtest.sh $< $(patsubst %.rob,%,$<)

# Compilar .rob para .o
out/%.o : %.rob ${ROBCMP} Makefile
	@mkdir -p out
	${ROBCMP} ${OPT} -I ../../lib $< -Oz -o $@

# Linkar .o com bibliotecas TFLM para criar executável
out/% : out/%.o
	${CLANG} -fuse-ld=lld \
	-flto=thin -fwhole-program-vtables \
	-Wl,--gc-sections -Wl,--as-needed -Wl,--strip-all \
	-Wl,--lto-whole-program-visibility \
	-o $@ $< ${WRAPPER_LIB} ${TFLM_LIB}

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
	@echo "  make          # Executa todos os testes"
	@echo "  make test     # Executa todos os testes"  
	@echo "  make list     # Lista testes disponíveis"
	@echo "  make clean    # Remove arquivos gerados"
	@echo "  make help     # Mostra esta ajuda"
	@echo ""
	@echo "Para executar teste individual:"
	@echo "  make arquivo.tt    # Executa apenas arquivo.rob"
	@echo ""
	@echo "Arquivos de teste encontrados:"
	@ls -1 *.rob 2>/dev/null || echo "  (nenhum arquivo .rob encontrado)"

.PHONY: all setup test clean list help