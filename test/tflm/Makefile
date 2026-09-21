# Build do tflite_info -- inspetor estático de modelos .tflite.
#
# Depende do submódulo third-party/tflite-micro e das dependências de terceiros
# que o TFLM baixa sob demanda (FlatBuffers). Se o build reclamar de
# flatbuffers/flatbuffers.h ausente, rode antes:
#
#   make -C ../../third-party/tflite-micro \
#        -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads

TFLM = ../../third-party/tflite-micro
DOWNLOADS = $(TFLM)/tensorflow/lite/micro/tools/make/downloads

CXX ?= clang++
CXXFLAGS = -std=c++17 -O2 -Wall
INCLUDES = -I$(TFLM) -I$(DOWNLOADS)/flatbuffers/include

# schema_utils.cc traz GetBuiltinCode(), que resolve builtin_code vs
# deprecated_builtin_code. Sem ele, modelos de conversores antigos reportam
# a lista de kernels errada, sem aviso.
SRCS = tflite_info.cpp $(TFLM)/tensorflow/compiler/mlir/lite/schema/schema_utils.cc

MODELS = $(wildcard models/*.tflite)

all: tflite_info

tflite_info: $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

# Inspeciona todos os modelos de validação de uma vez.
analyze: tflite_info
	@./tflite_info $(MODELS)

clean:
	rm -f tflite_info

.PHONY: all analyze clean
