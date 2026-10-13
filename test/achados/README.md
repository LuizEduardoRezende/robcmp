# Achados — demonstrações executáveis

Cada demonstração reproduz um problema encontrado durante a pesquisa e mostra o
resultado na tela. Rodar a partir desta pasta:

```bash
make
```

Ou uma de cada vez: `make signext`, `make use`, `make arena`.

Requisitos: o robcmp compilado (`build/robcmp`), `clang` e `gcc`; para a arena, a
`libtensorflow-microlite.a`, gerada por `make -C test/tflm check`.

---

## `make signext` — a RL passa −1 e a função C recebe 255

**Arquivos:** `signext/chama.rob` (RL) e `signext/amplia.c` (C).

A RL chama uma função C que recebe um `int8` e o devolve como `int32`, passando
−1. A função compilada pelo clang devolve **255**.

**Por quê.** Um `int8` ocupa 8 bits, mas é passado num registrador de 32 ou 64.
No LLVM IR, o atributo `signext` no parâmetro diz que quem chama já estendeu o
sinal, preenchendo os bits de cima com cópias do bit de sinal (−1 = `0xFF` vira
`0xFFFFFFFF`). O que a demonstração mostra, em três passos:

1. **IR:** o clang declara o parâmetro `i8 signext`; o robcmp declara `i8`, sem
   atributo — não promete nem faz a extensão.
2. **Código de máquina:** o clang confia na promessa e não estende — a função
   inteira é `mov %edi,%eax`. O gcc estende de novo por conta própria
   (`movsbl`).
3. **Execução:** com a função do clang, o −1 chega como `0x000000FF` = 255. Com
   a do gcc, chega certo.

O resultado depende de quem compilou o lado C, o que torna o erro difícil de
encontrar. No Cortex-M a convenção de chamada **obriga** quem chama a estender,
e o clang gera a função inteira como `bx lr`, sem extensão nenhuma.

**Causa no compilador:** `src/FunctionDecl.cpp` e `src/FunctionImpl.cpp` põem
`zeroext` nos tipos sem sinal e nada nos tipos com sinal — falta o `signext` para
`int8` e `int16`, em parâmetros e retornos. Hoje a API do wrapper não tem parâmetro `int8`/`int16`
escalar, então o TinyML não é afetado; a linguagem é. Correção prevista na
tarefa 3.8.

---

## `make use` — o modo padrão do `use` perde declarações

**Arquivos:** `use/<caso>/minhalib.rob` (a biblioteca) e `use/<caso>/main.rob`
(o programa que faz `use minhalib;`).

No modo padrão, o `use` lê a biblioteca por uma gramática simplificada
(`src/LanguageUse.y`), que guarda só as declarações e pula o corpo das funções.
Com `-bdep`, lê pela gramática completa e compila a biblioteca junto.

| Caso | O que a biblioteca tem | Sem `-bdep` | Com `-bdep` |
|---|---|---|---|
| `literal` | três funções; uma delas tem `a = {1.0, 2.0};` no corpo | **falha:** perde as três, até a que vem antes do literal | compila |
| `literal-controle` | igual, trocando o literal por `a = 1.0;` | compila | compila |
| `globais` | uma variável global e uma função | **falha:** a variável não é exportada | compila |

O caso de controle isola a causa: a mesma função, com corpo, funciona — o que
quebra é o literal de array. Os dois problemas já existiam antes desta pesquisa
(confirmado compilando o robcmp do commit `f6263b0`).

Por que importa: uma biblioteca em RL que use a sintaxe nativa terá
`m.input0 = {…}` no corpo das funções.

---

## `make arena` — quanto de arena cada modelo precisa

**Arquivo:** `arena/arena.cpp`, que roda o TFLM puro (sem o wrapper) sobre os 7
modelos de `test/tflm/models/`.

O `.tflite` **não guarda** o tamanho da arena: não há esse dado no arquivo. Os
números só existem rodando o TFLM sobre o modelo. Colunas:

| Coluna | O que é |
|---|---|
| usada | `arena_used_bytes()` depois do `AllocateTensors`: o que o TFLM informa |
| mínima alinhada | a menor arena com a qual o `AllocateTensors` passa, com o buffer começando num endereço múltiplo de 16 |
| mínima +8 bytes | a mesma medida, com o buffer começando 8 bytes depois |
| TCC | a arena que os testes do TCC reservavam |
| sobra no TCC | quanto o TCC reservava além da mínima alinhada |

O que a tabela mostra:

1. **O que o TFLM informa não é o mínimo.** No `hello_world`, ele usa 1216 bytes,
   mas só aloca com 1608: durante a alocação precisa de espaço temporário, que
   devolve no final. Um cálculo que reproduza só o "usado" subdimensiona a arena.
2. **O alinhamento do buffer muda a arena necessária.** O TFLM descarta os bytes
   até o próximo múltiplo de 16; com o buffer deslocado, todo modelo precisa de
   8 bytes a mais.
3. **O TCC reservava até 87% a mais** do que o necessário.

A mínima é obtida por busca binária, com cada tentativa num processo separado:
com arena pequena demais, o TFLM pode derrubar o processo em vez de devolver
erro. Números de x86_64; no Cortex-M, com ponteiros de 32 bits, devem ser
menores — não medido.
