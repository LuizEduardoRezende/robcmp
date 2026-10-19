// Referência de rob/nativo_efficientdet.rob: EfficientDet-Lite0 pelo TFLM
// direto, sem o wrapper, com a mesma imagem sintética
// do teste nativo. Imprime os elementos que o teste confere.
// make -C test/tflm ref_efficientdet
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
int main(int argc, char **argv) {
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), {});
    uint8_t *m = (uint8_t*)aligned_alloc(16, (b.size() + 15) / 16 * 16); memcpy(m, b.data(), b.size());
    size_t n = 10u << 20; uint8_t *arena = (uint8_t*)aligned_alloc(16, n);
    tflite::MicroMutableOpResolver<16> r;
    r.AddAdd(); r.AddConcatenation(); r.AddConv2D(); r.AddDepthwiseConv2D(); r.AddDequantize();
    r.AddLogistic(); r.AddMaxPool2D(); r.AddQuantize(); r.AddReshape(); r.AddResizeNearestNeighbor();
    tflite::MicroInterpreter it(tflite::GetModel(m), r, arena, n);
    if (it.AllocateTensors() != kTfLiteOk) return 1;
    TfLiteTensor *in = it.input(0);
    float scale = ((TfLiteAffineQuantization*)in->quantization.params)->scale->data[0];
    int zp = ((TfLiteAffineQuantization*)in->quantization.params)->zero_point->data[0];
    for (int i = 0; i < 307200; i++) {
        float v = (float)(i % 251) / 125.0f - 1.0f;       // gradiente em [-1, 1)
        long q = lroundf(v / scale) + zp;                  // mesma conversão do wrapper
        in->data.uint8[i] = (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q);
    }
    if (it.Invoke() != kTfLiteOk) return 2;
    float *scores = it.output(0)->data.f, *boxes = it.output(1)->data.f;
    float mn = 1e9, mx = -1e9;
    for (int i = 0; i < 1728540; i++) { mn = fminf(mn, scores[i]); mx = fmaxf(mx, scores[i]); }
    printf("scores: min %f max %f\n", mn, mx);
    int idx[] = {0, 1, 89, 90, 864270, 1728539};
    for (int k : idx) printf("score[%d] = %.6f\n", k, scores[k]);
    int bidx[] = {0, 1, 2, 3, 76823};
    for (int k : bidx) printf("box[%d] = %.6f\n", k, boxes[k]);
}
