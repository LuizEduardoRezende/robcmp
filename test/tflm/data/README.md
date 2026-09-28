# Dados de teste

Entradas dos testes portados do TCC (`../rob/tcc_*.rob`), em binário cru. O
`../rob_embed.py` os insere nos programas `.rob` durante o build.

| Arquivo | Bytes | Tipo | Origem |
|---|---|---|---|
| `person_image.bin` | 9216 | int8, 96×96×1 | `person_image_data.cc` gerado pelo build do TFLM (`tensorflow/lite/micro/examples/person_detection/testdata/person.bmp`) |
| `no_person_image.bin` | 9216 | int8, 96×96×1 | idem, `no_person.bmp` |
| `micro_speech_yes_features.bin` | 1960 | int8, 49×40 | vetor `dados_yes_input` de `test/tflm-tests/lib-micro-speech.rob`, branch `tcc/tflm-front-end` |
| `micro_speech_no_features.bin` | 1960 | int8, 49×40 | vetor `dados_no_input`, mesmo arquivo |
| `micro_speech_silence_features.bin` | 1960 | int8, 49×40 | vetor `dados_silence_input`, mesmo arquivo |

As duas imagens foram conferidas contra os vetores embutidos no teste do TCC
(`lib-person-detection.rob`): são idênticas byte a byte.

As features do micro speech são códigos int8 já na escala que o classificador
espera (scale 0,101716, zero_point −128). No TCC estavam declaradas como
`int16[]`; os valores cabem em int8 (mínimo −128, máximo 118).
