#!/usr/bin/env python3
"""
Embute arquivos binários num programa .rob, como vetores de bytes.

Uso:  rob_embed.py template.rob nome=arquivo[:int8] [...] > saida.rob

Cada `nome=arquivo` substitui a linha `// @nome@` do template pela declaração
`nome = {...};` com os bytes do arquivo. Serve para modelos .tflite e para
dados de teste (imagens, features).

Com `:int8`, os bytes são escritos como inteiros com sinal, em decimal
(-128 a 127), e o vetor é int8[] -- o tipo que a visão .raw de um tensor int8
exige na sintaxe nativa. O -128 vai como int8(-128).

Os vetores não vão para um módulo separado porque, nesta branch, `use` não
exporta variáveis globais (ver RESULTADOS.md, 5.4.14).

Sem `:int8`, os bytes são escritos em HEXADECIMAL de propósito: a RL infere
uint8[]. Em decimal sem sinal, um valor acima de 127 faz o literal ser inferido
como int16, e um vetor int16 passado a um parâmetro de outro tipo segue sem
conversão nem aviso (RESULTADOS.md, 5.4.9).
"""
import sys


def int8(b):
    # -128 é lido como -(128), e 128 é int16: um só -128 faria o vetor
    # inteiro int16[] (RESULTADOS.md, 5.2.20)
    return "int8(-128)" if b == 128 else "%d" % (b - 256 if b > 127 else b)


def declaracao(nome, caminho, com_sinal):
    dados = open(caminho, "rb").read()
    formato = int8 if com_sinal else (lambda b: "0x%02x" % b)
    linhas = ["    " + ", ".join(formato(b) for b in dados[i:i + 16])
              for i in range(0, len(dados), 16)]
    return "%s = {\n%s\n};" % (nome, ",\n".join(linhas))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    fonte = open(sys.argv[1], encoding="utf-8").read()
    for par in sys.argv[2:]:
        nome, _, caminho = par.partition("=")
        com_sinal = caminho.endswith(":int8")
        if com_sinal:
            caminho = caminho[:-len(":int8")]
        marcador = "// @%s@" % nome
        if not caminho or marcador not in fonte:
            sys.exit("erro: '%s' invalido ou marcador '%s' ausente em %s"
                     % (par, marcador, sys.argv[1]))
        fonte = fonte.replace(marcador, declaracao(nome, caminho, com_sinal), 1)
    sys.stdout.write(fonte)


if __name__ == "__main__":
    main()
