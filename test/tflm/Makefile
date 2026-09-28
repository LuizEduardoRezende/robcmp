# Ferramentas e validação do suporte a TinyML.
#
#   make              compila o tflite_info (inspetor estático de .tflite)
#   make analyze      inspeciona todos os modelos de models/
#   make check        roda a validação completa do wrapper (camadas 1 e 2)
#
# Depende do submódulo third-party/tflite-micro. Na primeira vez, `make check`
# compila a libtensorflow-microlite.a e baixa as dependências do TFLM (demora
# alguns minutos).

ROOT = ../..
TFLM = $(ROOT)/third-party/tflite-micro
DOWNLOADS = $(TFLM)/tensorflow/lite/micro/tools/make/downloads
OUT = out

# ---------------------------------------------------------------------------
# tflite_info
# ---------------------------------------------------------------------------

CXX ?= clang++
CXXFLAGS = -std=c++17 -O2 -Wall
INCLUDES = -I$(TFLM) -I$(DOWNLOADS)/flatbuffers/include

# schema_utils.cc traz GetBuiltinCode(), que resolve builtin_code vs
# deprecated_builtin_code. Sem ele, modelos de conversores antigos reportam
# a lista de kernels errada, sem aviso.
SRCS = tflite_info.cpp $(TFLM)/tensorflow/compiler/mlir/lite/schema/schema_utils.cc

MODELS = $(wildcard models/*.tflite)

all: tflite_info

tflite_info: $(SRCS) | $(DOWNLOADS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $@

# Inspeciona todos os modelos de validação de uma vez.
analyze: tflite_info
	@./tflite_info $(MODELS)

$(DOWNLOADS):
	$(MAKE) -C $(TFLM) -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads

# ---------------------------------------------------------------------------
# Validação do wrapper
# ---------------------------------------------------------------------------

ROBCMP = $(ROOT)/build/robcmp
GEN = $(TFLM)/gen/linux_x86_64_default_gcc
TFLM_LIB = $(GEN)/lib/libtensorflow-microlite.a
GENFILES = $(GEN)/genfiles
EXAMPLES = $(GENFILES)/tensorflow/lite/micro/examples

# O wrapper é compilado com g++, o mesmo compilador da biblioteca do TFLM.
WCXX = g++

# -DTF_LITE_STATIC_MEMORY é OBRIGATÓRIO. O TFLM é compilado com ele, e o define
# muda o layout de TfLiteTensor (64 bytes com, 112 sem). Um wrapper compilado
# sem ele lê data, type e params em offsets errados -- lixo silencioso, sem erro
# de compilação ou de linkedição.
WRAPPER_FLAGS = -std=c++17 -O2 -Wall -DTF_LITE_STATIC_MEMORY -fno-rtti -fno-exceptions \
                -ffunction-sections -fdata-sections
WRAPPER_INCLUDES = -I$(ROOT) -I$(TFLM) -I$(GENFILES) \
                   -I$(DOWNLOADS)/flatbuffers/include -I$(DOWNLOADS)/gemmlowp

# Dados de teste (imagens e áudio) gerados pelo build do TFLM.
TESTDATA = person_detection/testdata/person_image_data \
           person_detection/testdata/no_person_image_data \
           micro_speech/testdata/no_30ms_audio_data \
           micro_speech/testdata/yes_1000ms_audio_data \
           micro_speech/testdata/no_1000ms_audio_data
TESTDATA_OBJS = $(addprefix $(OUT)/testdata/,$(addsuffix .o,$(notdir $(TESTDATA))))

$(TFLM_LIB):
	$(MAKE) -C $(TFLM) -f tensorflow/lite/micro/tools/make/Makefile microlite -j$$(nproc)

$(OUT):
	mkdir -p $(OUT)/testdata

$(OUT)/libtflm_wrapper.a: $(ROOT)/wrappers/tflm/tflm_wrapper.cpp $(ROOT)/wrappers/tflm/tflm_wrapper.h | $(OUT) $(TFLM_LIB)
	$(WCXX) $(WRAPPER_FLAGS) $(WRAPPER_INCLUDES) -c $< -o $(OUT)/tflm_wrapper.o
	rm -f $@
	ar rcs $@ $(OUT)/tflm_wrapper.o

$(OUT)/testdata/%.o: | $(OUT) $(TFLM_LIB)
	$(WCXX) $(WRAPPER_FLAGS) $(WRAPPER_INCLUDES) -c $(filter %/$*.cc,$(addprefix $(EXAMPLES)/,$(addsuffix .cc,$(TESTDATA)))) -o $@

# Camada 1: o wrapper exercitado diretamente, comparado byte a byte com o TFLM.
$(OUT)/validate_wrapper: validate_wrapper.cpp $(TESTDATA_OBJS) $(OUT)/libtflm_wrapper.a
	$(WCXX) $(WRAPPER_FLAGS) $(WRAPPER_INCLUDES) validate_wrapper.cpp $(TESTDATA_OBJS) \
		$(OUT)/libtflm_wrapper.a $(TFLM_LIB) -Wl,--gc-sections -o $@

# Camada 2: programa RL compilado pelo robcmp para o host (sem -a) e linkado.
$(OUT)/hello_world_float.rob: rob/hello_world_float.rob models/hello_world_float.tflite tflite_to_rob.py | $(OUT)
	python3 tflite_to_rob.py models/hello_world_float.tflite modelo $< > $@

$(OUT)/hello_world_float: $(OUT)/hello_world_float.rob $(OUT)/libtflm_wrapper.a $(ROBCMP)
	$(ROBCMP) -I $(ROOT)/lib -o $(OUT)/hello_world_float.o $<
	$(WCXX) $(OUT)/hello_world_float.o $(OUT)/libtflm_wrapper.a $(TFLM_LIB) -Wl,--gc-sections -o $@

# Os testes leem os modelos por caminho relativo à raiz do repositório.
validate: $(OUT)/validate_wrapper
	@cd $(ROOT) && test/tflm/$(OUT)/validate_wrapper 2>/dev/null

rob: $(OUT)/hello_world_float
	@echo "== camada 2: programa RL (hello_world_float)"
	@$(OUT)/hello_world_float; st=$$?; \
	 if [ $$st -eq 0 ]; then echo "OK"; else echo "FALHOU na etapa $$st"; exit 1; fi

check: validate rob

clean:
	rm -rf tflite_info $(OUT)

.PHONY: all analyze validate rob check clean
