#!/usr/bin/env python3
"""
Embute arquivos binários num programa .rob, como vetores de bytes.

Uso:  rob_embed.py template.rob nome=arquivo [nome=arquivo ...] > saida.rob

Cada `nome=arquivo` substitui a linha `// @nome@` do template pela declaração
`nome = {...};` com os bytes do arquivo. Serve para modelos .tflite e para
dados de teste (imagens, features).

Os vetores não vão para um módulo separado porque, nesta branch, `use` não
exporta variáveis globais (ver RESULTADOS.md, 5.4.14).

Os bytes são escritos em HEXADECIMAL de propósito: a RL infere int8[]. Em
decimal, um valor acima de 127 faz o literal ser inferido como int16, e um
vetor int16 passado a um parâmetro de outro tipo segue sem conversão nem aviso
(RESULTADOS.md, 5.4.9).
"""
import sys


def declaracao(nome, caminho):
    dados = open(caminho, "rb").read()
    linhas = ["    " + ", ".join("0x%02x" % b for b in dados[i:i + 16])
              for i in range(0, len(dados), 16)]
    return "%s = {\n%s\n};" % (nome, ",\n".join(linhas))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    fonte = open(sys.argv[1], encoding="utf-8").read()
    for par in sys.argv[2:]:
        nome, _, caminho = par.partition("=")
        marcador = "// @%s@" % nome
        if not caminho or marcador not in fonte:
            sys.exit("erro: '%s' invalido ou marcador '%s' ausente em %s"
                     % (par, marcador, sys.argv[1]))
        fonte = fonte.replace(marcador, declaracao(nome, caminho), 1)
    sys.stdout.write(fonte)


if __name__ == "__main__":
    main()
