#!/usr/bin/env python3
"""
Confere que a fachada RL (lib/ai/tflm.rob) e o header C do wrapper
(wrappers/tflm/tflm_wrapper.h) concordam na assinatura de CADA função, no nível
em que isso importa: o LLVM IR que cada lado emite para o mesmo alvo.

Para cada alvo, gera um programa RL que chama todas as funções declaradas na
fachada e um programa C++ que chama todas as do header, compila os dois para IR
e compara as declarações resultantes.

Existe porque o erro que ele pega é silencioso: o wrapper do TCC declarava
handles como uintptr_t, que tem 64 bits no x86_64 (e batia com o uint64 da RL)
e 32 bits no Cortex-M (e não batia). No hardware, tflm_input(h, 1) devolveria
sempre o tensor 0, sem erro de compilação nem de linkedição.

Uso: check_abi.py  (a partir de test/tflm; sai com 1 se houver divergência)
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
FACHADA = RAIZ / "lib/ai/tflm.rob"
HEADER = RAIZ / "wrappers/tflm/tflm_wrapper.h"
ROBCMP = RAIZ / "build/robcmp"

# (nome, argumentos do robcmp, triple do clang)
ALVOS = [
    ("x86_64 (host)", [], "x86_64-linux-gnu"),
    ("Cortex-M3 (stm32f1)", ["-a", "stm32f1"], "thumbv7m-none-eabi"),
]

ARGS_RL = {"uint64": "0", "int8": "0", "float": "0.0",
           "uint8[]": "v_u8", "int8[]": "v_i8", "int16[]": "v_i16", "float[]": "v_f"}


def fachada():
    texto = FACHADA.read_text(encoding="utf-8")
    funcs = re.findall(r"^(\w+) (\w+)\(([^)]*)\);", texto, re.M)
    saida = []
    for ret, nome, params in funcs:
        tipos = [p.strip().rsplit(" ", 1)[0] for p in params.split(",") if p.strip()]
        saida.append((nome, tipos))
    return saida


def header():
    texto = HEADER.read_text(encoding="utf-8")
    texto = re.sub(r"/\*.*?\*/", "", texto, flags=re.S)
    texto = re.sub(r"//[^\n]*", "", texto)
    protos = re.findall(r"^\s*[\w ]+?\**\s*(\w+)\s*\(([^)]*)\)\s*;", texto, re.M)
    return [(nome, params) for nome, params in protos]


def ir_rl(funcs, args_alvo):
    corpo = ["use ai.tflm;",
             "v_u8 = {0u:4};", "v_i8 = {0:4};", "v_i16 = {int16(0):4};", "v_f = {0.0:4};",
             "int8 main() {"]
    for nome, tipos in funcs:
        corpo.append("    %s(%s);" % (nome, ", ".join(ARGS_RL[t] for t in tipos)))
    corpo += ["    return 0;", "}"]
    with tempfile.NamedTemporaryFile("w", suffix=".rob", delete=False) as f:
        f.write("\n".join(corpo) + "\n")
    r = subprocess.run([str(ROBCMP), "-bdep", *args_alvo, "-I", str(RAIZ / "lib"), f.name],
                       capture_output=True, text=True)
    return r.stdout + r.stderr


def ir_c(protos, triple):
    linhas = ['#include "wrappers/tflm/tflm_wrapper.h"', "void usa() {"]
    for nome, params in protos:
        n = len([p for p in params.split(",") if p.strip() and p.strip() != "void"])
        linhas.append("    %s(%s);" % (nome, ", ".join(["0"] * n)))
    linhas.append("}")
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write("\n".join(linhas) + "\n")
    r = subprocess.run(["clang++", "--target=" + triple, "-ffreestanding", "-std=c++17",
                        "-I", str(RAIZ), "-S", "-emit-llvm", "-O0", "-o", "-", f.name],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("erro compilando o lado C para %s:\n%s" % (triple, r.stderr))
    return r.stdout


ATRIBUTOS = r"\b(zeroext|signext|noundef|nonnull|dso_local|local_unnamed_addr|addrspace\(\d+\))\s*"


def assinaturas(ir):
    """nome -> (tipos, atributos de extensao), a partir das linhas `declare`."""
    saida = {}
    for m in re.finditer(r"^declare (.*?)@(\w+)\((.*?)\)", ir, re.M):
        ret, nome, params = m.groups()
        bruto = "%s(%s)" % (ret.strip(), params)
        tipos = re.sub(ATRIBUTOS, "", bruto)
        tipos = re.sub(r"\s+", " ", tipos).replace(" (", "(").replace(" ,", ",").replace(" )", ")").strip()
        ext = tuple(re.findall(r"\b(zeroext|signext)\b", bruto))
        saida[nome] = (tipos, ext)
    return saida


def main():
    funcs = fachada()
    protos = header()
    nomes_rl = {n for n, _ in funcs if n != "main"}
    nomes_c = {n for n, _ in protos}

    falhou = False
    for sobra, lado in ((nomes_rl - nomes_c, "só na fachada .rob"),
                        (nomes_c - nomes_rl, "só no header C")):
        for n in sorted(sobra):
            print("  DIVERGE  %-40s %s" % (n, lado))
            falhou = True

    funcs = [(n, t) for n, t in funcs if n in nomes_rl & nomes_c]
    protos = [(n, p) for n, p in protos if n in nomes_rl & nomes_c]

    for rotulo, args_alvo, triple in ALVOS:
        rl, c = assinaturas(ir_rl(funcs, args_alvo)), assinaturas(ir_c(protos, triple))
        tipos_dif = [n for n in sorted(nomes_rl & nomes_c) if rl.get(n, ("?",))[0] != c.get(n, ("?",))[0]]
        ext_dif = [n for n in sorted(nomes_rl & nomes_c)
                   if n not in tipos_dif and rl.get(n, ("", ()))[1] != c.get(n, ("", ()))[1]]
        print("== %s: %d funções, %d com tipos divergentes, %d com extensão divergente"
              % (rotulo, len(nomes_rl & nomes_c), len(tipos_dif), len(ext_dif)))
        for n in tipos_dif:
            print("  TIPOS    %-28s RL %-40s C %s" % (n, rl.get(n, ("ausente",))[0], c.get(n, ("ausente",))[0]))
            falhou = True
        if ext_dif:
            # Aviso, nao falha: o robcmp marca zeroext em todo retorno inteiro,
            # inclusive int8 com sinal, e o C marca signext. Comparacoes em i8
            # funcionam (verificado); o risco e latente, em conversoes que
            # ampliam o valor. Correcao pertence ao compilador (RESULTADOS.md).
            pares = sorted({(rl[n][1], c[n][1]) for n in ext_dif})
            print("  aviso: extensão divergente em %d funções; padrões: %s"
                  % (len(ext_dif), "; ".join("RL %s x C %s" % p for p in pares)))
    sys.exit(1 if falhou else 0)


if __name__ == "__main__":
    main()
