# Demonstração da sintaxe nativa de TinyML, para rodar ao vivo.
#
#   make -C test/tflm/demo exemplo   o hello world nativo: compila, roda, mostra as saídas
#   make -C test/tflm/demo ir        o que o compilador gerou a partir do .tflite
#   make -C test/tflm/demo erros     erros que a sintaxe nativa pega na compilação
#   make -C test/tflm/demo testes    os 5 testes do TCC nos dois modos
#   make -C test/tflm/demo memoria   onde o modelo fica no binário, em cada modo
#   make -C test/tflm/demo tudo      tudo acima, em ordem
#
# O roteiro da reunião, com estes comandos e as saídas esperadas, está em
# reuniao.html (abrir no navegador: xdg-open test/tflm/demo/reuniao.html).
#
# Usa o compilador de build/ e o wrapper e o TFLM de test/tflm (make -C test/tflm
# check os prepara).

ROOT = ../../..
ROBCMP = $(ROOT)/build/robcmp
TFLM_LIB = $(ROOT)/third-party/tflite-micro/gen/linux_x86_64_default_gcc/lib/libtensorflow-microlite.a
WRAPPER = ../out/libtflm_wrapper.a
OUT = out
LINK = g++ -Wl,--gc-sections

titulo = @printf "\n\033[1m== %s\033[0m\n" $(1)

$(OUT):
	@mkdir -p $(OUT)

$(WRAPPER):
	@$(MAKE) -s -C .. out/libtflm_wrapper.a

exemplo: | $(OUT) $(WRAPPER)
	$(call titulo,"seno.rob na sintaxe nativa: entrada x e as saídas dos modelos float e int8")
	@$(ROBCMP) -I $(ROOT)/lib -o $(OUT)/seno.o seno.rob
	@$(LINK) $(OUT)/seno.o $(WRAPPER) $(TFLM_LIB) -o $(OUT)/seno
	@$(OUT)/seno 2>&1 | paste - - - | awk '{printf "  x = %-9s float: %-10s int8: %s\n", $$1, $$2, $$3}'

ir: | $(OUT)
	$(call titulo,"gerado pelo compilador a partir do .tflite (IR antes da otimização)")
	@$(ROBCMP) -Od -I $(ROOT)/lib seno.rob 2>/dev/null > $(OUT)/seno.ll
	@grep -E '^@seno\.(tflite|arena)' $(OUT)/seno.ll | sed -E 's/c".*"/c"..."/; s/^/  /'
	@sed -n '/^define internal i64 @seno.ai_model/,/^}/p' $(OUT)/seno.ll | sed 's/^/  /'

erros: | $(OUT)
	$(call titulo,"erros de compilação: um programa por erro em erros/")
	@python3 -c "d=bytearray(open('../models/hello_world_float.tflite','rb').read()); d[-8:-4]=(15).to_bytes(4,'little'); d[-1]=15; open('erros/lsh.tflite','wb').write(d)"
	@for f in erros/*.rob; do \
		printf "\n  \033[1m%s\033[0m\n" $$f; sed -n '/^ *[a-z].*[;{]$$/p' $$f | grep -v "^int8 main\|return 0" | sed 's/^ */    /'; \
		$(ROBCMP) $$f 2>&1 >/dev/null | sed -E 's/\x1b\[[0-9;]*m//g' | grep "semantic error" | \
			sed -E "s#'[^']*/#'#g; s/^[^:]*:([0-9]+):([0-9]+): semantic error: /    -> linha \1, coluna \2: /"; \
	done
	@rm -f erros/lsh.tflite

testes:
	$(call titulo,"os 5 testes do TCC: modo biblioteca e sintaxe nativa")
	@$(MAKE) -s -C .. rob nativo 2>&1 | grep -E "==|OK|FALHOU"

memoria: | $(OUT) $(WRAPPER)
	$(call titulo,"o mesmo programa nos dois modos: seções do objeto")
	@python3 ../rob_embed.py lib_seno.rob modelo=../models/hello_world_float.tflite > $(OUT)/lib_seno.rob
	@$(ROBCMP) -I $(ROOT)/lib -o $(OUT)/lib_seno.o $(OUT)/lib_seno.rob
	@$(ROBCMP) -o $(OUT)/nat_seno.o nat_seno.rob
	@for m in lib nat; do printf "  %s_seno.o\n" $$m; \
		objdump -h $(OUT)/$${m}_seno.o | awk '/\.(data|rodata|bss)\./{printf "    %-20s %6d bytes\n", $$2, ("0x" $$3) + 0}' ; done 2>/dev/null || true
	$(call titulo,"person_detect: tamanho do executável (size)")
	@$(MAKE) -s -C .. out/tcc_person_detect out/nativo_person_detect > /dev/null
	@size ../out/tcc_person_detect ../out/nativo_person_detect | awk 'NR==1{printf "  %-10s %-10s %-10s %s\n", "text", "data", "bss", "programa"} NR>1{n=split($$6,p,"/"); printf "  %-10s %-10s %-10s %s\n", $$1, $$2, $$3, p[n]}'
	@echo "  (data ocupa flash E RAM num microcontrolador; o STM32F407 tem 192 KB de RAM)"

tudo: exemplo ir erros testes memoria

clean:
	rm -rf $(OUT) erros/lsh.tflite

.PHONY: exemplo ir erros testes memoria tudo clean
