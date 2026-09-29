#!/usr/bin/env python3
"""
Valida o módulo de análise de .tflite do compilador (src/tinyml/TfliteModel.cpp)
por meio do tflite_info, que é um executável fino sobre ele.

  1. Tabela de kernels: cada função tflm_add_* do header do wrapper tem
     exatamente uma entrada na tabela do módulo, e vice-versa. Builtins seguem a
     regra de nome (FULLY_CONNECTED -> tflm_add_fully_connected); customs usam o
     custom_code com que o resolver do TFLM os registra.
  2. Modelos reais: fatos conhecidos, incluindo a armadilha do builtin_code
     (RESULTADOS.md 5.2.4) no spam.tflite.
  3. Robustez: arquivo inexistente, arquivo que não é modelo, modelo truncado e
     modelo com bytes corrompidos precisam dar erro -- nunca derrubar o
     compilador nem ler fora do buffer. Roda sobre uma build com
     AddressSanitizer, para que uma leitura inválida apareça mesmo sem crash.

Uso: check_tflite.py TFLITE_INFO TFLITE_INFO_ASAN   (a partir de test/tflm)
"""
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
HEADER = RAIZ / "wrappers/tflm/tflm_wrapper.h"
RESOLVER = RAIZ / "third-party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
ETHOSU = RAIZ / "third-party/tflite-micro/tensorflow/lite/micro/kernels/ethos_u/ethosu.cc"
MODELOS = Path(__file__).resolve().parent / "models"

falhas = []


def falha(msg):
    falhas.append(msg)
    print(f"  FALHOU  {msg}")


def roda(binario, *args):
    return subprocess.run([binario, *map(str, args)], capture_output=True, text=True)


def checa_tabela(info):
    print("== tabela kernel -> função do wrapper x tflm_wrapper.h")
    tabela = [l.split() for l in roda(info, "--kernel-table").stdout.splitlines()]
    funcoes = [f for _, _, f in tabela]
    header = re.findall(r"\b(tflm_add_\w+)\s*\(", HEADER.read_text())

    for f in sorted(set(funcoes) - set(header)):
        falha(f"{f} está na tabela e não no header")
    for f in sorted(set(header) - set(funcoes)):
        falha(f"{f} está no header e não na tabela: kernel seria rejeitado como sem suporte")
    for f in sorted({f for f in funcoes if funcoes.count(f) > 1}):
        falha(f"{f} aparece mais de uma vez na tabela")

    customs_tflm = set(re.findall(r'AddCustom\(\s*"([^"]+)"', RESOLVER.read_text()))
    customs_tflm |= set(re.findall(r'GetString_ETHOSU\(\)\s*\{\s*return\s*"([^"]+)"', ETHOSU.read_text()))
    for tipo, op, f in tabela:
        if tipo == "builtin" and f != "tflm_add_" + op.lower():
            falha(f"builtin {op} mapeado para {f}, esperado tflm_add_{op.lower()}")
        if tipo == "custom" and op not in customs_tflm:
            falha(f"custom {op} não é um nome registrado pelo resolver do TFLM")

    print(f"  {len(tabela)} kernels na tabela, {len(set(header))} funções no header")


def kernels(saida):
    """{op: função do wrapper} a partir da saída do tflite_info."""
    return {op: f for _, op, f in
            re.findall(r"^\s+(builtin|custom)\s+(\S+)\s+(.+?)\s*$", saida, re.M)}


def checa_modelos(info):
    print("== modelos reais")
    esperado = {
        # builtin_code zerado: lido direto daria ADD ADD ADD ADD (5.2.4)
        "spam.tflite": {"kernels": {"GATHER", "MEAN", "FULLY_CONNECTED", "SOFTMAX"},
                        "linhas": ["input [0] tipo=int32    shape=[1,20] elementos=20"]},
        "hello_world_float.tflite": {
            "kernels": {"FULLY_CONNECTED"},
            "linhas": ["input [0] tipo=float32  shape=[1,1] elementos=1\n",
                       "output[0] tipo=float32  shape=[1,1] elementos=1\n"]},
        "hello_world_int8.tflite": {
            "kernels": {"FULLY_CONNECTED"},
            "linhas": ["scale=0.0244801 zero_point=-128", "scale=0.00829096 zero_point=5"]},
        "person_detect.tflite": {
            "kernels": {"AVERAGE_POOL_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "RESHAPE", "SOFTMAX"},
            "linhas": ["shape=[1,96,96,1] elementos=9216"]},
        "micro_speech_quantized.tflite": {
            "kernels": {"DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "RESHAPE", "SOFTMAX"},
            "linhas": ["shape=[1,1960] elementos=1960  scale=0.101716 zero_point=-128",
                       "shape=[1,4] elementos=4  scale=0.00390625 zero_point=-128"]},
        # os 9 custom ops do pipeline de áudio precisam ter suporte (5.2.5)
        "audio_preprocessor_float.tflite": {
            "kernels": {"CAST", "CONCATENATION", "MUL", "RESHAPE", "STRIDED_SLICE",
                        "SignalEnergy", "SignalFftAutoScale", "SignalFilterBank",
                        "SignalFilterBankLog", "SignalFilterBankSpectralSubtraction",
                        "SignalFilterBankSquareRoot", "SignalPCAN", "SignalRfft",
                        "SignalWindow"},
            "linhas": ["input [0] tipo=int16    shape=[1,480] elementos=480\n",
                       "output[0] tipo=float32  shape=[40] elementos=40\n"]},
        "audio_preprocessor_int8.tflite": {
            "kernels": {"ADD", "CAST", "CONCATENATION", "DIV", "MAXIMUM", "MINIMUM", "MUL",
                        "RESHAPE", "STRIDED_SLICE", "SignalEnergy", "SignalFftAutoScale",
                        "SignalFilterBank", "SignalFilterBankLog",
                        "SignalFilterBankSpectralSubtraction", "SignalFilterBankSquareRoot",
                        "SignalPCAN", "SignalRfft", "SignalWindow"},
            "linhas": ["output[0] tipo=int8     shape=[40] elementos=40\n"]},
    }
    # Todo modelo de models/ precisa ter fatos esperados aqui: um modelo novo
    # sem entrada faria o teste passar sem cobri-lo.
    for nome in sorted({m.name for m in MODELOS.glob("*.tflite")} - set(esperado)):
        falha(f"{nome} está em models/ e não tem fatos esperados neste teste")

    for nome, esp in esperado.items():
        r = roda(info, MODELOS / nome)
        if r.returncode != 0:
            falha(f"{nome}: {r.stderr.strip()}")
            continue
        ks = kernels(r.stdout)
        if set(ks) != esp["kernels"]:
            falha(f"{nome}: kernels {sorted(ks)}, esperado {sorted(esp['kernels'])}")
        for op, f in ks.items():
            if f == "SEM SUPORTE":
                falha(f"{nome}: kernel {op} sem suporte")
        for linha in esp["linhas"]:
            if linha not in r.stdout:
                falha(f"{nome}: saída não contém {linha.strip()!r}")
        print(f"  {nome:34} ok")


def checa_erro(asan, arquivo, rotulo, esperado=None):
    """Arquivo inválido: exit 1 com mensagem. Aceita também exit 0 quando a
    corrupção produz um modelo estruturalmente válido -- o que não pode é
    crash ou erro do ASan."""
    r = roda(asan, arquivo)
    if r.returncode < 0 or "AddressSanitizer" in r.stderr or r.returncode not in (0, 1):
        falha(f"{rotulo}: codigo {r.returncode}\n{r.stderr[-2000:]}")
        return None
    if esperado is not None and (r.returncode != 1 or esperado not in r.stderr):
        falha(f"{rotulo}: esperado erro contendo {esperado!r}, obtido codigo "
              f"{r.returncode} {r.stderr.strip()!r}")
    return r.returncode


def checa_robustez(asan):
    print("== robustez contra arquivos inválidos (AddressSanitizer)")
    base = (MODELOS / "hello_world_int8.tflite").read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        checa_erro(asan, tmp / "nao_existe.tflite", "inexistente", "file not found")
        checa_erro(asan, Path(__file__).resolve().parent / "Makefile", "não-modelo",
                   "TFL3 identifier missing")
        vazio = tmp / "vazio.tflite"
        vazio.write_bytes(b"")
        checa_erro(asan, vazio, "vazio", "TFL3 identifier missing")

        # Truncado em todos os comprimentos do cabeçalho e em pontos ao longo
        # do arquivo. Todo prefixo próprio precisa ser rejeitado.
        cortes = sorted(set(range(8, 64)) | set(range(64, len(base), 37)))
        rejeitados = 0
        for n in cortes:
            f = tmp / "truncado.tflite"
            f.write_bytes(base[:n])
            rejeitados += checa_erro(asan, f, f"truncado em {n} bytes",
                                     "corrupted or truncated") == 1
        print(f"  truncados: {len(cortes)} comprimentos, {rejeitados} rejeitados")

        # Bytes corrompidos. Alguns resultados ainda são modelos válidos; o
        # requisito é não haver crash nem leitura fora do buffer.
        rnd = random.Random(20260928)
        rejeitados = crashes = 0
        for i in range(300):
            dados = bytearray(base)
            for _ in range(rnd.randint(1, 8)):
                dados[rnd.randrange(8, len(dados))] = rnd.randrange(256)
            f = tmp / "corrompido.tflite"
            f.write_bytes(bytes(dados))
            r = checa_erro(asan, f, f"corrompido #{i}")
            rejeitados += r == 1
            crashes += r is None
        print(f"  corrompidos: 300 arquivos, {rejeitados} rejeitados, {crashes} crash(es)")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    info, asan = sys.argv[1], sys.argv[2]
    checa_tabela(info)
    checa_modelos(info)
    checa_robustez(asan)
    if falhas:
        print(f"{len(falhas)} falha(s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
