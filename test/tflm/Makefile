# Ferramentas e validação do suporte a TinyML.
#
#   make              compila o tflite_info (inspetor estático de .tflite)
#   make analyze      inspeciona todos os modelos de models/
#   make check        validação completa: wrapper (C++), testes do TCC em RL
#                     nos dois modos (ai.tflm e sintaxe nativa), assinaturas
#                     RL x C em x86_64 e Cortex-M, o módulo de análise de
#                     .tflite do compilador e o ai_model(...)
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
INCLUDES = -I$(ROOT)/src -I$(TFLM) -I$(DOWNLOADS)/flatbuffers/include

# O tflite_info é um executável fino sobre o módulo de análise do compilador:
# a mesma fonte, não uma segunda implementação.
SRCS = tflite_info.cpp $(ROOT)/src/tinyml/TfliteModel.cpp
DEPS = $(SRCS) $(ROOT)/src/tinyml/TfliteModel.h

MODELS = $(wildcard models/*.tflite)

all: tflite_info

tflite_info: $(DEPS) | $(DOWNLOADS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $@

# Mesmo módulo com AddressSanitizer, para o teste de robustez.
$(OUT)/tflite_info_asan: $(DEPS) | $(DOWNLOADS) $(OUT)
	$(CXX) $(CXXFLAGS) -g -fsanitize=address,undefined -fno-sanitize-recover=all \
		$(INCLUDES) $(SRCS) -o $@

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

# Camada 2: os testes do TCC portados para a API atual (rob/tcc_*.rob), em RL,
# compilados pelo robcmp para o host (sem -a) e linkados com o wrapper e o TFLM.
# O rob_embed.py insere nos templates os bytes dos modelos e dos dados de teste.
ROB_TESTS = tcc_hello_world_float tcc_hello_world_int8 tcc_spam tcc_person_detect tcc_micro_speech

EMBED = python3 rob_embed.py

$(OUT)/tcc_hello_world_float.rob: rob/tcc_hello_world.rob models/hello_world_float.tflite rob_embed.py | $(OUT)
	$(EMBED) $< modelo=models/hello_world_float.tflite > $@

$(OUT)/tcc_hello_world_int8.rob: rob/tcc_hello_world.rob models/hello_world_int8.tflite rob_embed.py | $(OUT)
	$(EMBED) $< modelo=models/hello_world_int8.tflite > $@

$(OUT)/tcc_spam.rob: rob/tcc_spam.rob models/spam.tflite rob_embed.py | $(OUT)
	$(EMBED) $< modelo=models/spam.tflite > $@

$(OUT)/tcc_person_detect.rob: rob/tcc_person_detect.rob models/person_detect.tflite data/person_image.bin data/no_person_image.bin rob_embed.py | $(OUT)
	$(EMBED) $< modelo=models/person_detect.tflite \
		imagem_pessoa=data/person_image.bin imagem_sem_pessoa=data/no_person_image.bin > $@

$(OUT)/tcc_micro_speech.rob: rob/tcc_micro_speech.rob models/micro_speech_quantized.tflite data/micro_speech_yes_features.bin data/micro_speech_no_features.bin data/micro_speech_silence_features.bin rob_embed.py | $(OUT)
	$(EMBED) $< modelo=models/micro_speech_quantized.tflite \
		features_yes=data/micro_speech_yes_features.bin \
		features_no=data/micro_speech_no_features.bin \
		features_silence=data/micro_speech_silence_features.bin > $@

# Camada 3: os mesmos testes na sintaxe nativa (rob/nativo_*.rob). O compilador
# lê o .tflite: só os dados de teste são embutidos, como int8[] para a .raw.
# Os modelos são citados por caminho relativo ao .rob gerado, em out/.
NATIVE_TESTS = nativo_hello_world_float nativo_hello_world_int8 nativo_spam \
               nativo_person_detect nativo_micro_speech nativo_simple_add \
               nativo_especificacao

$(OUT)/nativo_hello_world_float.rob: rob/nativo_hello_world.rob models/hello_world_float.tflite | $(OUT)
	sed 's/@modelo@/hello_world_float/' $< > $@

$(OUT)/nativo_hello_world_int8.rob: rob/nativo_hello_world.rob models/hello_world_int8.tflite | $(OUT)
	sed 's/@modelo@/hello_world_int8/' $< > $@

$(OUT)/nativo_spam.rob: rob/nativo_spam.rob models/spam.tflite | $(OUT)
	cp $< $@

$(OUT)/nativo_simple_add.rob: rob/nativo_simple_add.rob models/simple_add_model.tflite | $(OUT)
	cp $< $@

# A especificação inteira, compilada, ligada ao wrapper e executada: toda
# função que o compilador chama precisa existir no wrapper. Os modelos são
# citados relativos a test/tflm, por isso os caminhos são reescritos.
$(OUT)/nativo_especificacao.rob: native-syntax.rob $(MODELS) | $(OUT)
	sed 's#"models/#"../models/#' $< > $@

$(OUT)/nativo_person_detect.rob: rob/nativo_person_detect.rob models/person_detect.tflite data/person_image.bin data/no_person_image.bin rob_embed.py | $(OUT)
	$(EMBED) $< imagem_pessoa=data/person_image.bin:int8 \
		imagem_sem_pessoa=data/no_person_image.bin:int8 > $@

$(OUT)/nativo_micro_speech.rob: rob/nativo_micro_speech.rob models/micro_speech_quantized.tflite data/micro_speech_yes_features.bin data/micro_speech_no_features.bin data/micro_speech_silence_features.bin rob_embed.py | $(OUT)
	$(EMBED) $< features_yes=data/micro_speech_yes_features.bin:int8 \
		features_no=data/micro_speech_no_features.bin:int8 \
		features_silence=data/micro_speech_silence_features.bin:int8 > $@

$(addprefix $(OUT)/,$(ROB_TESTS) $(NATIVE_TESTS)): $(OUT)/%: $(OUT)/%.rob $(OUT)/libtflm_wrapper.a $(ROBCMP)
	$(ROBCMP) -I $(ROOT)/lib -o $(OUT)/$*.o $<
	$(WCXX) $(OUT)/$*.o $(OUT)/libtflm_wrapper.a $(TFLM_LIB) -Wl,--gc-sections -o $@

# Os testes leem os modelos por caminho relativo à raiz do repositório.
validate: $(OUT)/validate_wrapper
	@cd $(ROOT) && test/tflm/$(OUT)/validate_wrapper 2>/dev/null

# Roda cada teste de uma lista; o código de saída 0 é sucesso.
roda_testes = falhou=0; for t in $(1); do \
		$(OUT)/$$t > $(OUT)/$$t.log 2>&1; st=$$?; \
		if [ $$st -eq 0 ]; then printf "  %-28s OK\n" $$t; \
		else printf "  %-28s FALHOU (codigo %d)\n" $$t $$st; falhou=1; fi; \
	done; exit $$falhou

rob: $(addprefix $(OUT)/,$(ROB_TESTS))
	@echo "== camada 2: testes do TCC portados, em RL (ai.tflm)"
	@$(call roda_testes,$(ROB_TESTS))

nativo: $(addprefix $(OUT)/,$(NATIVE_TESTS))
	@echo "== camada 3: os mesmos testes na sintaxe nativa"
	@$(call roda_testes,$(NATIVE_TESTS))

# Confere, para x86_64 e Cortex-M, que cada funcao declarada em lib/ai/tflm.rob
# tem a mesma assinatura no LLVM IR que a do header C. Pega o tipo de erro que
# so apareceria no hardware (handle de 64 bits na RL, 32 bits no C).
abi:
	@echo "== assinaturas: lib/ai/tflm.rob x tflm_wrapper.h"
	@python3 check_abi.py

# Módulo de análise de .tflite do compilador: tabela de kernels x header,
# fatos dos modelos reais e robustez contra arquivos truncados/corrompidos.
tflite: tflite_info $(OUT)/tflite_info_asan
	@python3 check_tflite.py ./tflite_info $(OUT)/tflite_info_asan

# ai_model(...) na semântica do compilador: erros da seção 8 de native-syntax.rob
# que dependem só da declaração, busca do arquivo e retrocompatibilidade (7a).
ai_model: $(ROBCMP)
	@python3 check_ai_model.py $(ROBCMP)

check: validate rob nativo abi tflite ai_model

clean:
	rm -rf tflite_info $(OUT)

.PHONY: all analyze validate rob nativo abi tflite ai_model check clean
