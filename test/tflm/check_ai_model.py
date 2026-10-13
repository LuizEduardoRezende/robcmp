#!/usr/bin/env python3
"""
Valida o reconhecimento de ai_model(...) na análise semântica do compilador
(src/tinyml/AiModel.cpp), tarefa 3.3.

  1. Erros da seção 8 de native-syntax.rob que dependem só da declaração do
     modelo: caminho não literal, arena ausente ou inválida, arquivo
     inexistente, arquivo que não é .tflite e .tflite truncado. Cada um precisa
     sair na linha e coluna certas.
  2. Busca do arquivo: primeiro a pasta do .rob que faz a chamada, depois as
     pastas -I na ordem dada. A pasta de trabalho não entra.
  3. ai_model é nome reservado: declarar função, método, tipo, campo,
     variável, parâmetro, interface ou enum com ele é erro.
  4. Retrocompatibilidade (seção 7b): os nomes dos membros de um modelo não
     são reservados, e um tipo do usuário com input0, input() e output() segue
     compilando e rodando.

Enquanto a geração de código (3.6) não existe, um modelo aceito para na
geração com a mensagem "was read, but the code generation ... is not
implemented yet", que nomeia o arquivo lido. É ela que os casos de sucesso
conferem.

Uso: check_ai_model.py ROBCMP   (a partir de test/tflm)
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MODELOS = Path(__file__).resolve().parent / "models"
MODELO = MODELOS / "hello_world_float.tflite"
LIDO = "was read, but the code generation"

falhas = []


def falha(msg):
    falhas.append(msg)
    print(f"  FALHOU  {msg}")


def programa(*linhas):
    """Um main com as linhas dadas. A primeira linha do corpo é a linha 2."""
    corpo = "".join(f"    {l}\n" for l in linhas)
    return f"int16 main() {{\n{corpo}    return 0;\n}}\n"


def compila(robcmp, fonte, *args, cwd):
    r = subprocess.run([robcmp, *args, str(fonte)], cwd=cwd,
                       capture_output=True, text=True)
    return r.returncode, re.sub(r"\x1b\[[0-9;]*m", "", r.stderr)


def checa(rotulo, robcmp, fonte, esperado, linha, coluna=None, *args, cwd, arquivo=None):
    """Espera exatamente um erro, contendo `esperado`, em arquivo:linha:coluna.
    O arquivo do erro é, por padrão, o compilado."""
    arquivo = str(arquivo or fonte)
    codigo, saida = compila(robcmp, fonte, *args, cwd=cwd)
    erros = re.findall(r"^(.*?):(\d+):(\d+): semantic error: (.*)$", saida, re.M)
    if codigo == 0 or len(erros) != 1:
        falha(f"{rotulo}: esperado 1 erro, obtido codigo {codigo}: {saida.strip()!r}")
        return
    a, l, c, msg = erros[0]
    if (esperado not in msg or a != arquivo or int(l) != linha
            or (coluna is not None and int(c) != coluna)):
        local = f"{arquivo}:{linha}" + (f":{coluna}" if coluna is not None else "")
        falha(f"{rotulo}: esperado {esperado!r} em {local}, obtido {a}:{l}:{c} {msg!r}")
        return
    print(f"  ok  {rotulo}")


def checa_erros(robcmp, tmp):
    print("== erros de declaração (seção 8)")
    shutil.copy(MODELO, tmp / "m.tflite")
    shutil.copy(Path(__file__).resolve().parent / "Makefile", tmp / "Makefile")
    (tmp / "truncado.tflite").write_bytes(MODELO.read_bytes()[:500])

    casos = [
        # rótulo, linhas do main, mensagem esperada, linha, coluna
        ("caminho não literal",
         ['caminho = "m.tflite";', "a = ai_model(caminho, 3000);"],
         "must be a string literal", 3, 18),
        ("arena ausente", ['a = ai_model("m.tflite");'],
         "arena size of ai_model is missing", 2, 9),
        ("sem argumentos", ["a = ai_model();"], "called with 0", 2, 9),
        ("três argumentos", ['a = ai_model("m.tflite", 3000, 1);'], "called with 3", 2, 9),
        ("arena variável", ["n = 3000;", 'a = ai_model("m.tflite", n);'],
         "must be an integer constant", 3, 30),
        ("arena float", ['a = ai_model("m.tflite", 3000.0);'],
         "must be an integer constant", 2, 30),
        ("arena zero", ['a = ai_model("m.tflite", 0);'], "must be positive, not 0", 2, 30),
        ("arena negativa", ['a = ai_model("m.tflite", -5);'], "must be positive, not -5", 2, 31),
        ("arquivo inexistente", ['a = ai_model("nao_existe.tflite", 3000);'],
         "file not found", 2, 18),
        ("não é .tflite", ['a = ai_model("Makefile", 3000);'],
         "TFL3 identifier missing", 2, 18),
        ("truncado", ['a = ai_model("truncado.tflite", 3000);'],
         "corrupted or truncated", 2, 18),
        ("modelo aceito", ['a = ai_model("m.tflite", 3000);'], LIDO, 2, 9),
        ("arena constante nomeada", ["const ARENA = 3000;", 'a = ai_model("m.tflite", ARENA);'],
         LIDO, 3, 9),
        ("arena como expressão", ['a = ai_model("m.tflite", 90 * 1024);'], LIDO, 2, 9),
    ]
    for rotulo, linhas, esperado, linha, coluna in casos:
        fonte = tmp / "t.rob"
        fonte.write_text(programa(*linhas))
        checa(rotulo, robcmp, fonte, esperado, linha, coluna, cwd=tmp)


def checa_busca(robcmp, tmp):
    print("== busca do arquivo: pasta do .rob, depois -I na ordem")
    prog, inc1, inc2, trabalho = (tmp / d for d in ("prog", "inc1", "inc2", "trabalho"))
    for d in (prog, inc1, inc2, trabalho):
        d.mkdir()
    invalido = b"nao e um modelo"
    fonte = prog / "t.rob"
    fonte.write_text(programa('a = ai_model("m.tflite", 3000);'))

    def coloca(*pastas):
        for d in (prog, inc1, inc2, trabalho):
            (d / "m.tflite").unlink(missing_ok=True)
        for d, dados in pastas:
            (d / "m.tflite").write_bytes(dados)

    valido = MODELO.read_bytes()
    I = lambda *ds: [a for d in ds for a in ("-I", str(d))]

    # O caminho é relativo ao .rob, não à pasta de onde o robcmp é chamado.
    coloca((prog, valido))
    checa("ao lado do .rob, robcmp chamado de outra pasta", robcmp, fonte,
          f"model '{prog / 'm.tflite'}' {LIDO}", 2, cwd=trabalho)

    coloca((inc1, valido))
    checa("só na pasta -I", robcmp, fonte, f"model '{inc1 / 'm.tflite'}' {LIDO}", 2,
          None, *I(inc1), cwd=trabalho)

    coloca((prog, valido), (inc1, invalido))
    checa("pasta do .rob antes da -I", robcmp, fonte, f"model '{prog / 'm.tflite'}' {LIDO}", 2,
          None, *I(inc1), cwd=trabalho)

    coloca((inc1, valido), (inc2, invalido))
    checa("-I na ordem dada (primeira válida)", robcmp, fonte,
          f"model '{inc1 / 'm.tflite'}' {LIDO}", 2, None, *I(inc1, inc2), cwd=trabalho)
    checa("-I na ordem dada (primeira inválida)", robcmp, fonte,
          "TFL3 identifier missing", 2, None, *I(inc2, inc1), cwd=trabalho)

    # Diferente do `use`, a pasta de trabalho não é procurada.
    coloca((trabalho, valido))
    checa("pasta de trabalho não entra", robcmp, fonte, "file not found", 2, cwd=trabalho)

    # Chamada dentro de um arquivo de `use`, compilado junto (-bdep): o
    # caminho é relativo a esse arquivo, e o erro aponta para ele.
    lib = tmp / "lib" / "sub"
    lib.mkdir(parents=True)
    (lib / "m.tflite").write_bytes(valido)
    (lib / "carrega.rob").write_text(
        'int8 carrega() {\n    a = ai_model("m.tflite", 3000);\n    return 0;\n}\n')
    usa = prog / "usa.rob"
    usa.write_text("use sub.carrega;\nint16 main() {\n    carrega();\n    return 0;\n}\n")
    checa("chamada em arquivo de use", robcmp, usa, f"model '{lib / 'm.tflite'}' {LIDO}", 2, 9,
          "-bdep", *I(tmp / "lib"), cwd=trabalho, arquivo=lib / "carrega.rob")


def checa_reservado(robcmp, tmp):
    print("== ai_model é nome reservado")
    # A declaração nunca é a primeira do arquivo: o robcmp reporta a posição da
    # primeira declaração errada, 0:0 ou o início da linha (defeito do léxico e
    # do parser, RESULTADOS.md 2026-10-13).
    antes = "int16 f() {\n    return 0;\n}\n"
    main = "int16 main() {\n    return 0;\n}\n"
    casos = [
        # rótulo, declarações, linha, coluna
        ("função", "int16 ai_model(int16 v) {\n    return v;\n}\n", 4, 7),
        ("declaração de função", "int16 ai_model(int16 v);\n", 4, 7),
        ("método", "type t {\n    int8 ai_model() {\n        return 1;\n    }\n}\n", 5, 10),
        ("tipo", "type ai_model {\n    v = 1;\n}\n", 4, 6),
        ("campo", "type t {\n    ai_model = 1;\n}\n", 5, 5),
        ("variável global", "ai_model = 1;\n", 4, 1),
        ("parâmetro", "int16 g(int16 ai_model) {\n    return ai_model;\n}\n", 4, 9),
        ("parâmetro de método",
         "type t {\n    int8 m(int8 ai_model) {\n        return ai_model;\n    }\n}\n", 5, 12),
        ("interface", "interface ai_model {\n    int8 f();\n}\n", 4, 11),
        ("enum", "enum ai_model {A, B}\n", 4, 6),
    ]
    fonte = tmp / "r.rob"
    for rotulo, declaracao, linha, coluna in casos:
        fonte.write_text(antes + declaracao + main)
        checa(rotulo, robcmp, fonte, "ai_model is a reserved name", linha, coluna, cwd=tmp)
    fonte.write_text(programa("ai_model = 1;"))
    checa("variável local", robcmp, fonte, "ai_model is a reserved name", 2, 5, cwd=tmp)


def checa_retrocompatibilidade(robcmp, tmp):
    print("== retrocompatibilidade (seção 7b)")
    cc = os.environ.get("CC", "cc")
    rotulo = "tipo do usuário com input0, input() e output()"
    fonte = tmp / "r.rob"
    fonte.write_text(
        "type sensor {\n    input0 = 0;\n"
        "    int8 input() {\n        return 1;\n    }\n"
        "    int8 output() {\n        return 2;\n    }\n}\n"
        "int16 main() {\n    s = sensor();\n    s.input0 = 5;\n"
        "    if s.input() == 1 and s.output() == 2 and s.input0 == 5 {\n"
        "        return 0;\n    }\n    return 1;\n}\n")
    codigo, saida = compila(robcmp, fonte, "-o", str(tmp / "r.o"), cwd=tmp)
    if codigo != 0:
        falha(f"{rotulo}: não compilou: {saida.strip()!r}")
        return
    ligacao = subprocess.run([cc, str(tmp / "r.o"), "-o", str(tmp / "r")],
                             capture_output=True, text=True)
    if ligacao.returncode != 0:
        falha(f"{rotulo}: não ligou: {ligacao.stderr.strip()!r}")
        return
    execucao = subprocess.run([str(tmp / "r")])
    if execucao.returncode != 0:
        falha(f"{rotulo}: rodou com código {execucao.returncode}, esperado 0")
        return
    print(f"  ok  {rotulo}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    robcmp = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for etapa in (checa_erros, checa_busca, checa_reservado,
                      checa_retrocompatibilidade):
            pasta = tmp / etapa.__name__
            pasta.mkdir()
            etapa(robcmp, pasta)
    if falhas:
        print(f"{len(falhas)} falha(s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
