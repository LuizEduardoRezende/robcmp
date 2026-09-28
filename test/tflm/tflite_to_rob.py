#!/usr/bin/env python3
"""
Embute um arquivo .tflite num programa .rob, como vetor de bytes.

Uso:  tflite_to_rob.py modelo.tflite nome_variavel template.rob > saida.rob

O template precisa conter a linha `// @MODELO@`, que é substituída pela
declaração do vetor. O vetor não vai para um módulo separado porque, nesta
branch, `use` não exporta variáveis globais (ver RESULTADOS.md).

Os bytes são escritos em HEXADECIMAL de propósito. Em decimal, um literal com
valor acima de 127 é inferido como int16, e o vetor int16 é passado a um
parâmetro uint8[] sem conversão nem aviso -- o modelo chega corrompido.
"""
import sys

MARCADOR = "// @MODELO@"


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    caminho, nome, template = sys.argv[1:]

    dados = open(caminho, "rb").read()
    linhas = ["    " + ", ".join("0x%02x" % b for b in dados[i:i + 16])
              for i in range(0, len(dados), 16)]
    declaracao = "%s = {\n%s\n};" % (nome, ",\n".join(linhas))

    fonte = open(template, encoding="utf-8").read()
    if MARCADOR not in fonte:
        sys.exit("erro: %s nao contem a linha '%s'" % (template, MARCADOR))
    sys.stdout.write(fonte.replace(MARCADOR, declaracao, 1))


if __name__ == "__main__":
    main()
