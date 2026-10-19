# Modelos externos

Modelos públicos maiores que os de validação, baixados em 2026-10-19 para
levantar quantos tipos de kernel modelos reais usam (RESULTADOS.md, 5.4.25) e
para testes futuros. Não fazem parte da validação automática (`make -C
test/tflm check`): o `check_tflite.py` só confere os modelos de `models/`.

Nenhum cabe no STM32F407 (512 KB de flash): servem para testes no x86_64.

| Arquivo | Modelo | Tamanho | Entradas | Saídas | Kernels | Roda no TFLM |
|---|---|---|---|---|---|---|
| `mobilebert_classifier.tflite` | classificador de texto do MediaPipe, baseado no MobileBERT (Transformer) | 25,7 MB | 3 × int32[128] | float32[2] | 14 | sim |
| `efficientdet_lite0_int8.tflite` | detector de objetos EfficientDet-Lite0, int8 | 4,6 MB | uint8[320×320×3] | float32[19206×90], float32[19206×4] | 10 | sim |
| `ssd_mobilenet_v1_quant.tflite` | detector de objetos SSD MobileNet v1 (COCO), quantizado, com pós-processamento | 4,2 MB | uint8[300×300×3] | 4 saídas: caixas, classes, scores, contagem | 6 (1 custom) | sim, ver abaixo |
| `yamnet.tflite` | classificador de sons YAMNet (521 classes) | 4,1 MB | float32[15600] (0,975 s a 16 kHz) | float32[521] | 16 | **não**: `RFFT2D` e `COMPLEX_ABS` |

`ssd_mobilenet_v1_labelmap.txt` é a lista das classes do COCO que acompanha o
detector SSD.

**Saídas do SSD sem tamanho no arquivo.** As 4 saídas vêm do custom op
`TFLite_Detection_PostProcess`, que só define o tamanho delas em execução. No
`.tflite` elas não têm forma (`shape=[]`), e o compilador as tipa como um
elemento só. Usar esse modelo na sintaxe nativa exige tratar saídas de
tamanho desconhecido (RESULTADOS.md, 5.4.25).

**YAMNet.** O compilador recusa o modelo, nomeando os dois kernels que o TFLM
não implementa.

## Origem

Todos de servidores oficiais do Google. Os do MediaPipe estão descritos em
https://developers.google.com/edge/mediapipe/solutions (licença: ver a
página de cada modelo); o SSD vem do pacote de exemplos do TensorFlow Lite
(Apache 2.0).

| Arquivo | URL | SHA-256 |
|---|---|---|
| `mobilebert_classifier.tflite` | https://storage.googleapis.com/mediapipe-models/text_classifier/bert_classifier/float32/latest/bert_classifier.tflite | `9b45012ab143d88d61e10ea501d6c8763f7202b86fa987711519d89bfa2a88b1` |
| `efficientdet_lite0_int8.tflite` | https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite | `0720bf247bd76e6594ea28fa9c6f7c5242be774818997dbbeffc4da460c723bb` |
| `yamnet.tflite` | https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite | `4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf` |
| `ssd_mobilenet_v1_quant.tflite` | https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip (`detect.tflite`) | `e4b118e5e4531945de2e659742c7c590f7536f8d0ed26d135abcfe83b4779d13` |

Os endereços `latest` do MediaPipe podem passar a apontar para outra versão;
o SHA-256 identifica a baixada aqui.
