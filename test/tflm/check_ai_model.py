#!/usr/bin/env python3
"""
Valida ai_model(...) no compilador (src/tinyml): a análise semântica
(tarefas 3.3 a 3.5) e a geração de código (3.6).

  1. Erros da seção 8 de native-syntax.rob que dependem só da declaração do
     modelo: caminho não literal, arena ausente ou inválida, arquivo
     inexistente, arquivo que não é .tflite, .tflite truncado e kernel sem
     suporte no wrapper. Cada um precisa sair na linha e coluna certas.
  2. Busca do arquivo: primeiro a pasta do .rob que faz a chamada, depois as
     pastas -I na ordem dada. A pasta de trabalho não entra.
  3. ai_model é nome reservado: declarar função, método, tipo, campo,
     variável, parâmetro, interface ou enum com ele é erro.
  4. Retrocompatibilidade (seção 7b): os nomes dos membros de um modelo não
     são reservados, e um tipo do usuário com input0, input() e output() segue
     compilando e rodando.
  5. Membros do modelo (tarefa 3.4): native-syntax.rob inteiro passa pela
     análise semântica; as propriedades são constantes com os valores do
     .tflite; as duas visões têm os tipos das seções 3 e 4b; e cada erro das
     seções 3, 4b, 4c e 8 que depende só dos membros sai na posição certa.
  6. Tamanhos: vetor de tamanho conhecido escrito no tensor inteiro, índice
     constante de elemento e tensor lido para uma variável que já existe, nas
     duas visões (seções 4b e 8); vetor de tamanho desconhecido e índice
     variável ficam para o wrapper.

Um programa aceito precisa compilar sem erro, o que inclui o verificador do
LLVM sobre o IR gerado. Quantas vezes um modelo foi embutido se confere no IR
(-Od): um @<variável>.tflite por declaração. A execução dos modelos em sintaxe
nativa é testada à parte, em `make -C test/tflm nativo`.

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


def checa_ok(rotulo, robcmp, fonte, *args, cwd, modelos=None):
    """Compila sem erro nenhum. Com `modelos`, confere também quantos modelos o
    IR embute."""
    codigo, saida = compila(robcmp, fonte, *args, cwd=cwd)
    if codigo != 0:
        falha(f"{rotulo}: não compilou: {saida.strip()!r}")
        return
    if modelos is not None:
        r = subprocess.run([robcmp, "-Od", *args, str(fonte)], cwd=cwd,
                           capture_output=True, text=True)
        embutidos = re.findall(r"^@\"?([\w.]+)\.tflite\"? = ", r.stdout, re.M)
        if len(embutidos) != modelos:
            falha(f"{rotulo}: esperados {modelos} modelos embutidos, obtidos {embutidos!r}")
            return
    print(f"  ok  {rotulo}")


def kernel_sem_suporte(destino):
    """O hello_world_float com o operador trocado por LSH_PROJECTION, que o
    TFLM não implementa. Os 8 últimos bytes do arquivo são o OperatorCode do
    FULLY_CONNECTED (9): builtin_code (int32), padding e deprecated_builtin_code."""
    dados = bytearray(MODELO.read_bytes())
    assert dados[-8:] == bytes([9, 0, 0, 0, 0, 0, 0, 9])
    dados[-8:-4] = (15).to_bytes(4, "little")
    dados[-1] = 15
    destino.write_bytes(bytes(dados))


def checa_erros(robcmp, tmp):
    print("== erros de declaração (seção 8)")
    shutil.copy(MODELO, tmp / "m.tflite")
    shutil.copy(Path(__file__).resolve().parent / "Makefile", tmp / "Makefile")
    (tmp / "truncado.tflite").write_bytes(MODELO.read_bytes()[:500])
    kernel_sem_suporte(tmp / "lsh.tflite")

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
        ("kernel sem suporte", ['a = ai_model("lsh.tflite", 3000);'],
         "uses the builtin kernel LSH_PROJECTION, which the TFLM wrapper does not support",
         2, 18),
    ]
    fonte = tmp / "t.rob"
    for rotulo, linhas, esperado, linha, coluna in casos:
        fonte.write_text(programa(*linhas))
        checa(rotulo, robcmp, fonte, esperado, linha, coluna, cwd=tmp)

    aceitos = [
        ("modelo aceito", ['a = ai_model("m.tflite", 3000);']),
        ("arena constante nomeada", ["const ARENA = 3000;", 'a = ai_model("m.tflite", ARENA);']),
        ("arena como expressão", ['a = ai_model("m.tflite", 90 * 1024);']),
    ]
    for rotulo, linhas in aceitos:
        fonte.write_text(programa(*linhas))
        checa_ok(rotulo, robcmp, fonte, cwd=tmp, modelos=1)


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

    # Em cada caso, só um arquivo válido pode ser encontrado: os outros
    # candidatos não existem ou são inválidos. Compilar sem erro mostra que a
    # busca chegou a ele sem passar por um inválido.

    # O caminho é relativo ao .rob, não à pasta de onde o robcmp é chamado.
    coloca((prog, valido))
    checa_ok("ao lado do .rob, robcmp chamado de outra pasta", robcmp, fonte, cwd=trabalho)

    coloca((inc1, valido))
    checa_ok("só na pasta -I", robcmp, fonte, *I(inc1), cwd=trabalho)

    coloca((prog, valido), (inc1, invalido))
    checa_ok("pasta do .rob antes da -I", robcmp, fonte, *I(inc1), cwd=trabalho)

    coloca((inc1, valido), (inc2, invalido))
    checa_ok("-I na ordem dada (primeira válida)", robcmp, fonte, *I(inc1, inc2), cwd=trabalho)
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
    checa_ok("chamada em arquivo de use", robcmp, usa, "-bdep", *I(tmp / "lib"), cwd=trabalho)
    # e o erro de um arquivo de `use` aponta para ele
    (lib / "m.tflite").write_bytes(invalido)
    checa("erro em arquivo de use", robcmp, usa, "TFL3 identifier missing", 2, 18,
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


# Os modelos dos casos de membros, declarados nas linhas 2 a 5 do main.
MODELOS_MEMBROS = [
    ("d", "hello_world_float", 3000),         # float32, sem quantização
    ("q", "hello_world_int8", 3000),          # int8 quantizado, zero_point -128
    ("sp", "spam", 4096),                     # entrada int32
    ("pre", "audio_preprocessor_int8", 12288),  # entrada int16, saída int8 sem escala
]


def programa_membros(*linhas):
    """Um main com os quatro modelos declarados; a primeira linha dada é a 6.
    A função f, depois do main, recebe um valor qualquer."""
    decl = [f'{v} = ai_model("{n}.tflite", {a});' for v, n, a in MODELOS_MEMBROS]
    return programa(*decl, *linhas) + "int16 f(int16 v) {\n    return v;\n}\n"


def checa_unico(rotulo, robcmp, fonte, esperado, linha, coluna, cwd):
    """Exatamente um erro, na linha e coluna dadas."""
    codigo, saida = compila(robcmp, fonte, cwd=cwd)
    erros = re.findall(r"^.*?:(\d+):(\d+): semantic error: (.*)$", saida, re.M)
    if len(erros) != 1:
        falha(f"{rotulo}: esperado 1 erro, obtido {saida.strip()!r}")
        return
    l, c, msg = erros[0]
    if esperado not in msg or (int(l), int(c)) != (linha, coluna):
        falha(f"{rotulo}: esperado {esperado!r} em {linha}:{coluna}, obtido {l}:{c} {msg!r}")
        return
    print(f"  ok  {rotulo}")


def checa_membros(robcmp, tmp):
    print("== membros do modelo (3.4)")

    # A especificação inteira: o único erro é o do modelo hipotético de várias
    # entradas da seção 5, que não existe em models/ (tarefa 3.9).
    raiz = Path(__file__).resolve().parent
    codigo, saida = compila(robcmp, raiz / "native-syntax.rob", cwd=raiz)
    erros = re.findall(r"semantic error: (.*)$", re.sub(r"\x1b\[[0-9;]*m", "", saida), re.M)
    outros = [e for e in erros if "algum_modelo_multi_entrada" not in e]
    if outros or not erros:
        falha(f"native-syntax.rob: erros além dos esperados: {outros!r}")
    else:
        print("  ok  native-syntax.rob passa pela análise semântica")

    for _, nome, _ in MODELOS_MEMBROS:
        shutil.copy(MODELOS / f"{nome}.tflite", tmp / f"{nome}.tflite")
    fonte = tmp / "m.rob"

    # Propriedades são constantes de compilação: valem como arena, e a
    # verificação da arena mostra o valor -- zero_point -128, size 1,
    # input_count 1, size 40 da saída do audio_preprocessor.
    constantes = [
        ("zero_point é constante", "q.input0.zero_point", "not -128"),
        ("size é constante", "q.output0.size - 1", "not 0"),
        ("input_count é constante", "q.input_count - 1", "not 0"),
        ("size de tensor grande", "pre.output0.size - 40", "not 0"),
    ]
    for rotulo, expr, esperado in constantes:
        fonte.write_text(programa_membros(f'x = ai_model("hello_world_int8.tflite", {expr});'))
        checa_unico(rotulo, robcmp, fonte, esperado, 6, 45, tmp)

    # Tipos das visões, expostos pela mensagem de tipo errado.
    tipos = [
        ("leitura padrão é float[]", ["v = q.output0;", "q.input0.raw = v;"],
         "takes int8[], the type of the tensor, not float[]", 7, 5),
        (".raw de saída int8 é int8[]", ["c = pre.output0.raw;", "pre.input0.raw = c;"],
         "takes int16[], the type of the tensor, not int8[]", 7, 5),
        ("escrita padrão pede float[]", ["v = {int8(1):1};", "q.input0 = v;"],
         "takes a vector of real values, float[], not int8[]", 7, 5),
        (".raw exige o tipo exato", ["q.input0.raw = {int16(0):1};"],
         "takes int8[], the type of the tensor, not int16[]", 6, 5),
    ]
    for rotulo, linhas, esperado, linha, coluna in tipos:
        fonte.write_text(programa_membros(*linhas))
        checa_unico(rotulo, robcmp, fonte, esperado, linha, coluna, tmp)

    erros = [
        ("tensor inexistente", "d.input1 = {1.0};", "has 1 input tensor (input0): input1 does not exist", 5),
        ("tensor inexistente, elemento", "x = d.input1[0];", "input1 does not exist", 9),
        ("tensor inexistente, escrita de elemento", "d.input1[0] = 1.0;", "input1 does not exist", 5),
        ("scale somente leitura", "q.input0.scale = 0.5;", "q.input0.scale is read-only", 5),
        ("input_count somente leitura", "q.input_count = 2;", "q.input_count is read-only", 5),
        ("float32 sem scale", "s = d.input0.scale;", "is not quantized: it has no scale", 9),
        ("int32 sem zero_point", "z = sp.input0.zero_point;", "is not quantized: it has no zero_point", 9),
        ("saída int8 sem escala", "s = pre.output0.scale;", "is not quantized: it has no scale", 9),
        ("saída somente leitura", "d.output0 = {1.0};", "is an output tensor: it is read-only", 5),
        ("elemento de saída", "d.output0[0] = 1.0;", "is an output tensor: it is read-only", 5),
        ("saída .raw", "q.output0.raw = {int8(0):1};", "is an output tensor: it is read-only", 5),
        ("modelo copiado", "x = d;", "d is a model, not a value", 9),
        ("modelo passado a função", "f(d);", "d is a model, not a value", 7),
        ("modelo declarado de novo", 'd = ai_model("hello_world_int8.tflite", 3000);', "d already exists", 5),
        ("modelo atribuído", "d = 1;", "d already holds a model", 5),
        (".raw de float32", "d.input0.raw = {1};", "is float32: it has no raw view", 5),
        (".raw de int32", "v = sp.input0.raw;", "is int32: it has no raw view", 9),
        ("membro inexistente", "x = d.foo;", "Model d has no member foo", 9),
        ("membro de tensor inexistente", "x = d.input0.foo;", "has no member foo", 9),
        ("método inexistente", "d.run();", "Model d has no member run", 5),
        ("invoke com argumento", "d.invoke(1);", "d.invoke() takes no arguments", 5),
        ("invoke sem chamar", "x = d.invoke;", "call it as d.invoke()", 9),
        ("propriedade indexada", "x = d.input0.size[0];", "d.input0.size is not a vector", 9),
        ("tensor chamado", "x = d.input0();", "the only method of a model is invoke()", 9),
        ("ai_model passado a função", 'x = f(ai_model("hello_world_int8.tflite", 3000));',
         "it is used only to initialize a new variable", 11),
        ("ai_model solto", 'ai_model("hello_world_int8.tflite", 3000);',
         "it is used only to initialize a new variable", 5),
    ]
    for rotulo, linha_rob, esperado, coluna in erros:
        fonte.write_text(programa_membros(linha_rob))
        checa_unico(rotulo, robcmp, fonte, esperado, 6, coluna, tmp)

    # Modelo como campo de tipo: fora da especificação; recusado até a 3.6
    # decidir como guardar modelo e arena por instância.
    rotulo = "modelo como campo de tipo"
    fonte.write_text('type robo {\n    m = ai_model("hello_world_int8.tflite", 3000);\n}\n'
                     'int16 main() {\n    return 0;\n}\n')
    codigo, saida = compila(robcmp, fonte, cwd=tmp)
    outros = re.findall(r"^.*?:(\d+):(\d+): semantic error: (.*)$", saida, re.M)
    if len(outros) != 1 or outros[0][:2] != ("2", "5") or "field of a type" not in outros[0][2]:
        falha(f"{rotulo}: obtido {saida.strip()!r}")
    else:
        print(f"  ok  {rotulo}")

    # Ordem das declarações: um modelo global vale em todo o arquivo, como
    # qualquer global, mesmo declarado depois da função que o usa; um local
    # só vale depois da sua declaração.
    usa = ('int8 a() {\n    m.input0 = {1.0};\n    return m.invoke();\n}\n'
           'int8 b() {\n    m.input0[0] = 1.0;\n    return 0;\n}\n')
    declara = 'm = ai_model("hello_world_int8.tflite", 3000);\n'
    main = 'int16 main() {\n    return a() + b();\n}\n'
    fonte.write_text(usa + declara + main)
    checa_ok("global declarado depois das funções que o usam, embutido uma vez", robcmp, fonte,
             cwd=tmp, modelos=1)
    fonte.write_text(usa.replace("m.input0[0]", "m.input1[0]") + declara + main)
    saida_ordem = compila(robcmp, fonte, cwd=tmp)[1]
    outros = re.findall(r"^.*?:(\d+):(\d+): semantic error: (.*)$", saida_ordem, re.M)
    if (len(outros) != 1 or outros[0][:2] != ("6", "5")
            or "input1 does not exist" not in outros[0][2]):
        falha(f"erro de membro em global declarado depois: obtido {saida_ordem.strip()!r}")
    else:
        print("  ok  erro de membro em global declarado depois, uma vez só")

    fonte.write_text(programa("m.input0 = {1.0};", declara.strip()))
    codigo, saida = compila(robcmp, fonte, cwd=tmp)
    outros = re.findall(r"^.*?:(\d+):(\d+): semantic error: (.*)$", saida, re.M)
    if len(outros) != 1 or outros[0][:2] != ("2", "5") or "used before its declaration" not in outros[0][2]:
        falha(f"local usado antes da declaração: obtido {saida.strip()!r}")
    else:
        print("  ok  local usado antes da declaração")


def checa_tamanhos(robcmp, tmp):
    print("== tamanhos e índices (seções 4b e 8)")
    for _, nome, _ in MODELOS_MEMBROS:
        shutil.copy(MODELOS / f"{nome}.tflite", tmp / f"{nome}.tflite")
    fonte = tmp / "t.rob"

    # sp: entrada 20, saída 2; pre: entrada int16 480, saída int8 40
    erros = [
        ("literal maior que o tensor", ["d.input0 = {1.0, 2.0};"],
         "d.input0 has 1 element; the vector has 2.", 6, 5),
        ("literal na .raw", ["q.input0.raw = {int8(0), int8(0)};"],
         "q.input0.raw has 1 element; the vector has 2.", 6, 5),
        ("variável que guarda um literal", ["v = {0.0:3};", "d.input0 = v;"],
         "d.input0 has 1 element; the vector has 3.", 7, 5),
        ("outro tensor lido inteiro", ["sp.input0 = pre.output0;"],
         "sp.input0 has 20 elements; the vector has 40.", 6, 5),
        ("índice além do fim", ["x = sp.output0[2];"],
         "sp.output0 has 2 elements: index 2 is out of the tensor.", 6, 9),
        ("índice negativo", ["x = sp.output0[-1];"], "index -1 is out of the tensor", 6, 9),
        ("índice constante nomeado", ["const K = 2;", "x = sp.output0[K];"],
         "index 2 is out of the tensor", 7, 9),
        ("escrita de elemento na .raw", ["pre.input0.raw[480] = int16(0);"],
         "pre.input0.raw has 480 elements: index 480 is out of the tensor.", 6, 5),
        ("leitura de elemento na .raw", ["x = pre.output0.raw[40];"],
         "index 40 is out of the tensor", 6, 9),
        ("leitura para variável de outro tamanho", ["saida = {0.0:3};", "saida = sp.output0;"],
         "sp.output0 has 2 elements, and saida, which receives it, has 3.", 7, 5),
        ("leitura .raw para variável de outro tipo", ["c = {int16(0):40};", "c = pre.output0.raw;"],
         "pre.output0.raw is int8[], and c, which receives it, is int16[].", 7, 5),
        ("leitura para variável escalar", ["x = 1;", "x = sp.output0;"],
         "sp.output0 is float[], and x, which receives it, is int8.", 7, 5),
    ]
    for rotulo, linhas, esperado, linha, coluna in erros:
        fonte.write_text(programa_membros(*linhas))
        checa_unico(rotulo, robcmp, fonte, esperado, linha, coluna, tmp)

    aceitos = [
        ("literal do tamanho do tensor", ["sp.input0 = {0.0:20};"]),
        ("último elemento", ["x = sp.output0[1];", "pre.input0.raw[479] = int16(0);"]),
        ("índice variável", ["i = 7;", "x = pre.output0[i];"]),
        ("tensor lido inteiro, mesmo tamanho", ["v = q.output0;", "d.input0 = v;"]),
        ("leitura para buffer reservado", ["saida = {0.0:2};", "saida = sp.output0;",
                                            "c = {int8(0):40};", "c = pre.output0.raw;"]),
        ("buffer relido de outro modelo, mesmo tamanho", ["s = d.output0;", "s = q.output0;"]),
        ("leitura no laço", ["i = 0;", "while i < 3 {", "    s = sp.output0;", "    i = i + 1;", "}"]),
    ]
    for rotulo, linhas in aceitos:
        fonte.write_text(programa_membros(*linhas))
        checa_ok(rotulo, robcmp, fonte, cwd=tmp)

    fonte.write_text('d = ai_model("hello_world_float.tflite", 3000);\n'
                     'int8 escreve(float[] v) {\n    d.input0 = v;\n    return d.invoke();\n}\n'
                     'int16 main() {\n    v = {1.0};\n    return escreve(v);\n}\n')
    checa_ok("vetor de tamanho desconhecido (parâmetro)", robcmp, fonte, cwd=tmp)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    robcmp = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for etapa in (checa_erros, checa_busca, checa_reservado,
                      checa_retrocompatibilidade, checa_membros, checa_tamanhos):
            pasta = tmp / etapa.__name__
            pasta.mkdir()
            etapa(robcmp, pasta)
    if falhas:
        print(f"{len(falhas)} falha(s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
