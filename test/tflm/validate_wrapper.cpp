/*
 * validate_wrapper -- camada 1 da validação do wrapper TFLM.
 *
 * Exercita o wrapper diretamente, sem o robcmp no meio: se algo falhar aqui, o
 * defeito está no wrapper, não no compilador.
 *
 * Critério principal: TESTE DIFERENCIAL. Para a mesma entrada, o wrapper e o
 * TFLM usado diretamente (referência) precisam produzir exatamente os mesmos
 * bytes de saída. Isso separa "o wrapper está correto" de "o modelo acerta",
 * que é verificado à parte, como sanidade.
 *
 * Os dados de teste (imagens, áudio) vêm dos arquivos gerados pelo build do
 * TFLM em gen/.../genfiles. Rode `make microlite` no submódulo antes.
 */

#include "wrappers/tflm/tflm_wrapper.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/micro/examples/person_detection/testdata/no_person_image_data.h"
#include "tensorflow/lite/micro/examples/person_detection/testdata/person_image_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/no_1000ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/no_30ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/yes_1000ms_audio_data.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Infraestrutura
// ---------------------------------------------------------------------------

static int g_falhas = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            printf("  FALHOU: " __VA_ARGS__);             \
            printf("\n");                                 \
            g_falhas++;                                   \
        }                                                 \
    } while (0)

static std::vector<uint8_t> carregar(const char *path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> v((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    if (v.empty()) printf("  ERRO: nao foi possivel ler %s\n", path);
    return v;
}

// Arenas separadas para wrapper e referência: os dois precisam coexistir.
alignas(16) static uint8_t g_arena_wrapper[160 * 1024];
alignas(16) static uint8_t g_arena_ref[160 * 1024];
alignas(16) static uint8_t g_arena_aux[40 * 1024];

// O handle de tensor do wrapper é um TfLiteTensor*. Só o código de teste se
// apoia nisso, para ler e escrever bytes crus e comparar com a referência.
static TfLiteTensor *tensor_de(uintptr_t h) {
    return reinterpret_cast<TfLiteTensor *>(h);
}

static bool mesmos_bytes(const TfLiteTensor *a, const TfLiteTensor *b) {
    return a->bytes == b->bytes && memcmp(a->data.raw, b->data.raw, a->bytes) == 0;
}

// Referência: o mesmo modelo rodando no TFLM direto, sem o wrapper.
struct Referencia {
    tflite::MicroMutableOpResolver<32> resolver;
    tflite::MicroInterpreter *interp = nullptr;
    alignas(tflite::MicroInterpreter) uint8_t storage[sizeof(tflite::MicroInterpreter)];

    bool iniciar(const uint8_t *modelo, uint8_t *arena, size_t arena_size) {
        interp = new (storage) tflite::MicroInterpreter(
            tflite::GetModel(modelo), resolver, arena, arena_size);
        return interp->AllocateTensors() == kTfLiteOk;
    }
    ~Referencia() { if (interp) interp->~MicroInterpreter(); }
};

// ---------------------------------------------------------------------------
// 1. hello_world -- varredura da senoide
// ---------------------------------------------------------------------------

static void testa_senoide(const char *path) {
    printf("== senoide: %s\n", path);
    std::vector<uint8_t> modelo = carregar(path);
    if (modelo.empty()) { g_falhas++; return; }

    uintptr_t r = tflm_create_resolver();
    CHECK(r != 0, "tflm_create_resolver devolveu 0");
    CHECK(tflm_add_fully_connected(r) == 0, "tflm_add_fully_connected falhou");
    uintptr_t m = tflm_create_model(r, modelo.data(), g_arena_wrapper,
                                    (int32_t)modelo.size(), 3000);
    CHECK(m != 0, "tflm_create_model devolveu 0");
    if (!m) return;

    uintptr_t in = GetInputTensor(m, 0), out = GetOutputTensor(m, 0);
    float erro_max = 0.0f;
    const int N = 40;
    for (int i = 0; i <= N; ++i) {
        float x = 2.0f * (float)M_PI * i / N;
        SetTensorValue(in, 0, x);
        CHECK(InvokeInterpreter(m) == 0, "InvokeInterpreter falhou em x=%f", x);
        float e = fabsf(GetTensorAsFloat(out, 0) - sinf(x));
        if (e > erro_max) erro_max = e;
    }
    printf("  erro maximo contra sin(x) em %d pontos: %.4f\n", N + 1, erro_max);
    CHECK(erro_max < 0.25f, "erro maximo %.4f acima da tolerancia 0.25", erro_max);
    tflm_free_model(m);
}

// ---------------------------------------------------------------------------
// 2. Ciclo de vida
// ---------------------------------------------------------------------------

static void testa_ciclo_de_vida() {
    printf("== ciclo de vida\n");
    std::vector<uint8_t> modelo = carregar("test/tflm/models/hello_world_float.tflite");

    // Slots precisam ser reciclados: mais ciclos do que slots existentes.
    for (int i = 0; i < 10; ++i) {
        uintptr_t r = tflm_create_resolver();
        CHECK(r != 0, "sem slot livre na iteracao %d -- in_use nao resetado", i);
        if (!r) return;
        tflm_add_fully_connected(r);
        uintptr_t m = tflm_create_model(r, modelo.data(), g_arena_wrapper,
                                        (int32_t)modelo.size(), 3000);
        CHECK(m != 0, "create_model falhou na iteracao %d", i);
        tflm_free_model(m);
    }

    // Com todos os slots ocupados, o próximo falha de forma controlada.
    uintptr_t h[4];
    for (int i = 0; i < 4; ++i) h[i] = tflm_create_resolver();
    CHECK(tflm_create_resolver() == 0, "o quinto resolver deveria falhar");
    tflm_free_model(h[2]);
    uintptr_t outro = tflm_create_resolver();
    CHECK(outro != 0, "slot liberado nao foi reaproveitado");
    for (int i = 0; i < 4; ++i) if (i != 2) tflm_free_model(h[i]);
    tflm_free_model(outro);

    // Handles inválidos não podem derrubar o programa.
    CHECK(tflm_add_fully_connected(0) == -1, "add com handle 0 deveria devolver -1");
    CHECK(tflm_create_model(0, nullptr, g_arena_wrapper, 0, 0) == 0,
          "create_model com handle 0 deveria devolver 0");
    CHECK(InvokeInterpreter(0) == -1, "invoke com handle 0 deveria devolver -1");
    CHECK(GetInputTensor(0, 0) == 0, "GetInputTensor com handle 0 deveria devolver 0");
    tflm_free_model(0);

    // Arena pequena demais. Cada tamanho roda num processo filho, porque o
    // proprio TFLM cai em certas faixas (AllocatePOD constroi os parametros do
    // operador sobre nullptr quando a arena acaba -- ver RESULTADOS.md). O
    // wrapper so consegue garantir a faixa mais baixa, a das estruturas de
    // controle; as janelas de crash acima dela sao limitacao do TFLM e sao
    // reportadas, nao contadas como falha do wrapper.
    const int passo = 8, limite = 4096;
    int menor_ok = -1, crashes = 0, ultimo = -1, ini_faixa = 0;
    const char *estado[] = {"falha controlada", "OK", "CRASH no TFLM"};
    for (int tam = 0; tam <= limite; tam += passo) {
        pid_t p = fork();
        if (p == 0) {
            fclose(stderr);
            uintptr_t r = tflm_create_resolver();
            tflm_add_fully_connected(r);
            _exit(tflm_create_model(r, modelo.data(), g_arena_wrapper,
                                    (int32_t)modelo.size(), tam) ? 1 : 0);
        }
        int st;
        waitpid(p, &st, 0);
        int res = WIFSIGNALED(st) ? 2 : WEXITSTATUS(st);
        if (res == 1 && menor_ok < 0) menor_ok = tam;
        if (res == 2) crashes++;
        if (res != ultimo) {
            if (ultimo >= 0) printf("    arena %5d..%-5d %s\n", ini_faixa, tam - passo, estado[ultimo]);
            ini_faixa = tam;
            ultimo = res;
        }
    }
    printf("    arena %5d..%-5d %s\n", ini_faixa, limite, estado[ultimo]);
    CHECK(menor_ok > 0, "nenhum tamanho de arena funcionou");
    if (crashes)
        printf("  AVISO: %d tamanhos de arena derrubam o TFLM (limitacao conhecida, nao do wrapper)\n",
               crashes);
    // API crua: recusa tipo e tamanho errados, sem copiar nada.
    {
        uintptr_t r = tflm_create_resolver();
        tflm_add_fully_connected(r);
        uintptr_t m = tflm_create_model(r, modelo.data(), g_arena_wrapper,
                                        (int32_t)modelo.size(), 3000);
        uintptr_t in = GetInputTensor(m, 0);          // float32[1,1]
        int8_t b8[1] = {7};
        int16_t b16[1] = {7};
        CHECK(SetTensorRawInt8(in, b8, 1) == -1, "SetTensorRawInt8 aceitou tensor float32");
        CHECK(SetTensorRawInt16(in, b16, 1) == -1, "SetTensorRawInt16 aceitou tensor float32");
        CHECK(SetTensorRawInt8(0, b8, 1) == -1, "SetTensorRawInt8 aceitou handle 0");
        tflm_free_model(m);

        std::vector<uint8_t> q = carregar("test/tflm/models/hello_world_int8.tflite");
        r = tflm_create_resolver();
        tflm_add_fully_connected(r);
        m = tflm_create_model(r, q.data(), g_arena_wrapper, (int32_t)q.size(), 3000);
        in = GetInputTensor(m, 0);                    // int8[1,1]
        int8_t dois[2] = {1, 2};
        tensor_de(in)->data.int8[0] = 99;
        CHECK(SetTensorRawInt8(in, dois, 2) == -1, "SetTensorRawInt8 aceitou tamanho errado");
        CHECK(tensor_de(in)->data.int8[0] == 99, "recusa por tamanho ainda assim copiou dados");
        CHECK(SetTensorRawInt8(in, b8, 1) == 0, "SetTensorRawInt8 recusou entrada valida");
        CHECK(tensor_de(in)->data.int8[0] == 7, "SetTensorRawInt8 nao escreveu o valor");
        int8_t lido[1] = {0};
        CHECK(GetTensorRawInt8(in, lido, 1) == 0 && lido[0] == 7, "GetTensorRawInt8 nao leu o valor");

        // SetTensorFromIntArray copia CRU (comportamento do TCC), nao quantiza:
        // num tensor int8 com scale 0,0244801, o valor 5 tem de chegar como 5.
        const int16_t cru[1] = {5};
        SetTensorFromIntArray(in, cru, 1, 1);
        CHECK(tensor_de(in)->data.int8[0] == 5,
              "SetTensorFromIntArray quantizou (%d) em vez de copiar cru (5)",
              tensor_de(in)->data.int8[0]);
        tflm_free_model(m);
    }
    printf("  reciclagem, limite, handles nulos, arena insuficiente e recusas da API crua: ok\n");
}

// ---------------------------------------------------------------------------
// 3. spam -- entrada int32, saída float32
// ---------------------------------------------------------------------------

static void testa_spam() {
    printf("== spam\n");
    std::vector<uint8_t> modelo = carregar("test/tflm/models/spam.tflite");
    if (modelo.empty()) { g_falhas++; return; }

    // Frases do TCC (comparacao-firmwares/firmware_spam_detector.cc).
    const int32_t legitima[20] = {1, 82, 70, 214, 55, 98, 84, 47, 89, 26, 73};  // "This is a great article thank you for the information"
    const int32_t spam[20]     = {1, 93, 326, 72, 8, 485, 24, 35, 6};           // "win free money online click www site com"

    uintptr_t r = tflm_create_resolver();
    tflm_add_fully_connected(r);
    tflm_add_gather(r);
    tflm_add_mean(r);
    tflm_add_softmax(r);
    uintptr_t m = tflm_create_model(r, modelo.data(), g_arena_wrapper,
                                    (int32_t)modelo.size(), 16 * 1024);
    CHECK(m != 0, "tflm_create_model devolveu 0");
    if (!m) return;

    Referencia ref;
    ref.resolver.AddFullyConnected();
    ref.resolver.AddGather();
    ref.resolver.AddMean();
    ref.resolver.AddSoftmax();
    CHECK(ref.iniciar(modelo.data(), g_arena_ref, 16 * 1024), "referencia nao inicializou");

    const int32_t *frases[2] = {legitima, spam};
    const char *nomes[2] = {"legitima", "spam"};
    uintptr_t in = GetInputTensor(m, 0), out = GetOutputTensor(m, 0);
    for (int f = 0; f < 2; ++f) {
        for (int j = 0; j < 20; ++j) SetTensorValue(in, j, (float)frases[f][j]);
        memcpy(ref.interp->input(0)->data.i32, frases[f], 20 * sizeof(int32_t));

        CHECK(mesmos_bytes(tensor_de(in), ref.interp->input(0)),
              "entrada '%s' difere da referencia", nomes[f]);
        CHECK(InvokeInterpreter(m) == 0, "invoke falhou");
        CHECK(ref.interp->Invoke() == kTfLiteOk, "invoke da referencia falhou");
        CHECK(mesmos_bytes(tensor_de(out), ref.interp->output(0)),
              "saida '%s' difere da referencia", nomes[f]);

        float p_legit = GetTensorAsFloat(out, 0), p_spam = GetTensorAsFloat(out, 1);
        printf("  %-9s p(legitima)=%.4f p(spam)=%.4f  soma=%.4f\n",
               nomes[f], p_legit, p_spam, p_legit + p_spam);
        CHECK(fabsf(p_legit + p_spam - 1.0f) < 1e-3f, "softmax nao soma 1");
        if (f == 0) CHECK(p_legit > p_spam, "frase legitima classificada como spam");
        else        CHECK(p_spam > p_legit, "frase de spam classificada como legitima");
    }
    tflm_free_model(m);
}

// ---------------------------------------------------------------------------
// 4. person_detect -- 5 kernels, entrada int8 de 9216 elementos
// ---------------------------------------------------------------------------

static void testa_person_detect() {
    printf("== person_detect\n");
    std::vector<uint8_t> modelo = carregar("test/tflm/models/person_detect.tflite");
    if (modelo.empty()) { g_falhas++; return; }
    const int32_t arena = 136 * 1024;  // valor do teste oficial do TFLM

    uintptr_t r = tflm_create_resolver();
    tflm_add_average_pool_2d(r);
    tflm_add_conv_2d(r);
    tflm_add_depthwise_conv_2d(r);
    tflm_add_reshape(r);
    tflm_add_softmax(r);
    uintptr_t m = tflm_create_model(r, modelo.data(), g_arena_wrapper,
                                    (int32_t)modelo.size(), arena);
    CHECK(m != 0, "tflm_create_model devolveu 0");
    if (!m) return;

    Referencia ref;
    ref.resolver.AddAveragePool2D();
    ref.resolver.AddConv2D();
    ref.resolver.AddDepthwiseConv2D();
    ref.resolver.AddReshape();
    ref.resolver.AddSoftmax();
    CHECK(ref.iniciar(modelo.data(), g_arena_ref, arena), "referencia nao inicializou");

    uintptr_t in = GetInputTensor(m, 0), out = GetOutputTensor(m, 0);
    CHECK(GetTensorSize(in) == 9216, "entrada deveria ter 9216 elementos, tem %zu",
          GetTensorSize(in));

    // As imagens já estão em int8 quantizado. Pela API do wrapper, que só fala
    // float, é preciso desquantizar para a conversão reconstruir o mesmo byte.
    float scale = tensor_de(in)->params.scale;
    int32_t zp = tensor_de(in)->params.zero_point;

    const unsigned char *imagens[2] = {g_person_image_data, g_no_person_image_data};
    const char *nomes[2] = {"com pessoa", "sem pessoa"};
    std::vector<float> buf(9216);
    for (int k = 0; k < 2; ++k) {
        for (int i = 0; i < 9216; ++i)
            buf[i] = scale * ((float)(int8_t)imagens[k][i] - (float)zp);
        SetTensorArray(in, buf.data(), 9216, 9216);
        memcpy(ref.interp->input(0)->data.int8, imagens[k], 9216);

        CHECK(mesmos_bytes(tensor_de(in), ref.interp->input(0)),
              "imagem '%s': conversao float->int8 nao reconstruiu os bytes originais", nomes[k]);

        // O mesmo pela API crua: a imagem ja esta em int8, entra como esta.
        CHECK(SetTensorRawInt8(in, (const int8_t *)imagens[k], 9216) == 0,
              "SetTensorRawInt8 recusou a imagem");
        CHECK(mesmos_bytes(tensor_de(in), ref.interp->input(0)),
              "imagem '%s' pela API crua difere da referencia", nomes[k]);
        CHECK(InvokeInterpreter(m) == 0, "invoke falhou");
        CHECK(ref.interp->Invoke() == kTfLiteOk, "invoke da referencia falhou");
        CHECK(mesmos_bytes(tensor_de(out), ref.interp->output(0)),
              "saida '%s' difere da referencia", nomes[k]);

        // Índices do modelo: 0 = sem pessoa, 1 = pessoa.
        float nao = GetTensorAsFloat(out, 0), sim = GetTensorAsFloat(out, 1);
        printf("  %-10s p(pessoa)=%.4f p(sem pessoa)=%.4f\n", nomes[k], sim, nao);
        if (k == 0) CHECK(sim > nao, "pessoa nao detectada na imagem com pessoa");
        else        CHECK(nao > sim, "pessoa detectada na imagem sem pessoa");
    }
    tflm_free_model(m);
}

// ---------------------------------------------------------------------------
// 5. micro_speech -- preprocessor (9 custom ops) + classificador
// ---------------------------------------------------------------------------

static const int kAmostrasPorJanela = 480;  // 30 ms a 16 kHz
static const int kPassoAmostras = 320;      // 20 ms
static const int kFeatureSize = 40;
static const int kFeatureCount = 49;

static void registra_preprocessor(uintptr_t r) {
    tflm_add_reshape(r);
    tflm_add_cast(r);
    tflm_add_strided_slice(r);
    tflm_add_concatenation(r);
    tflm_add_mul(r);
    tflm_add_add(r);
    tflm_add_div(r);
    tflm_add_minimum(r);
    tflm_add_maximum(r);
    tflm_add_window(r);
    tflm_add_fft_auto_scale(r);
    tflm_add_rfft(r);
    tflm_add_energy(r);
    tflm_add_filter_bank(r);
    tflm_add_filter_bank_square_root(r);
    tflm_add_filter_bank_spectral_subtraction(r);
    tflm_add_pcan(r);
    tflm_add_filter_bank_log(r);
}

// Uma janela de áudio -> 40 features, pela API crua do wrapper: as amostras
// int16 entram como estão, e as features int8 saem como estão.
static bool extrai_feature(uintptr_t pre, const int16_t *audio, int8_t *saida) {
    uintptr_t in = GetInputTensor(pre, 0), out = GetOutputTensor(pre, 0);
    if (SetTensorRawInt16(in, audio, kAmostrasPorJanela) != 0) return false;
    if (InvokeInterpreter(pre) != 0) return false;
    return GetTensorRawInt8(out, saida, kFeatureSize) == 0;
}

static const char *kRotulos[4] = {"silence", "unknown", "yes", "no"};

static void testa_micro_speech() {
    printf("== micro_speech\n");
    std::vector<uint8_t> m_pre = carregar("test/tflm/models/audio_preprocessor_int8.tflite");
    std::vector<uint8_t> m_cls = carregar("test/tflm/models/micro_speech_quantized.tflite");
    if (m_pre.empty() || m_cls.empty()) { g_falhas++; return; }

    uintptr_t rp = tflm_create_resolver();
    registra_preprocessor(rp);
    uintptr_t pre = tflm_create_model(rp, m_pre.data(), g_arena_aux,
                                      (int32_t)m_pre.size(), sizeof(g_arena_aux));
    CHECK(pre != 0, "preprocessor nao inicializou");

    uintptr_t rc = tflm_create_resolver();
    tflm_add_reshape(rc);
    tflm_add_fully_connected(rc);
    tflm_add_depthwise_conv_2d(rc);
    tflm_add_softmax(rc);
    uintptr_t cls = tflm_create_model(rc, m_cls.data(), g_arena_wrapper,
                                      (int32_t)m_cls.size(), 40 * 1024);
    CHECK(cls != 0, "classificador nao inicializou");
    if (!pre || !cls) return;

    // 5a. Preprocessor bit a bit, contra o vetor esperado do teste oficial
    //     (micro_speech_test.cc, NoFeatureTest). Valida os 9 custom ops.
    const int8_t esperado[kFeatureSize] = {
        126, 103, 124, 102, 124, 102, 123, 100, 118, 97, 118, 100, 118, 98,
        121, 100, 121, 98,  117, 91,  96,  74,  54,  87, 100, 87,  109, 92,
        91,  80,  64,  55,  83,  74,  74,  78,  114, 95, 101, 81,
    };
    int8_t obtido[kFeatureSize];
    CHECK(extrai_feature(pre, g_no_30ms_audio_data, obtido), "invoke do preprocessor falhou");
    int diferentes = 0;
    for (int i = 0; i < kFeatureSize; ++i) diferentes += (obtido[i] != esperado[i]);
    printf("  preprocessor 'no_30ms': %d de %d features identicas ao esperado\n",
           kFeatureSize - diferentes, kFeatureSize);
    CHECK(diferentes == 0, "features do preprocessor divergem do teste oficial");

    // 5b. Pipeline completo. As features saem do preprocessor como int8 SEM
    //     parâmetros de quantização, e o classificador as espera como int8 COM
    //     scale/zero_point. Os códigos já estão na escala certa: o exemplo
    //     oficial os copia crus. Testamos os dois caminhos.
    uintptr_t in = GetInputTensor(cls, 0), out = GetOutputTensor(cls, 0);
    struct { const char *rotulo; const int16_t *audio; unsigned n; } amostras[2] = {
        {"yes", g_yes_1000ms_audio_data, g_yes_1000ms_audio_data_size},
        {"no",  g_no_1000ms_audio_data,  g_no_1000ms_audio_data_size},
    };
    static int8_t features[kFeatureCount * kFeatureSize];
    for (auto &a : amostras) {
        memset(features, 0, sizeof(features));
        unsigned pos = 0;
        for (int k = 0; k < kFeatureCount && pos + kAmostrasPorJanela <= a.n; ++k, pos += kPassoAmostras)
            extrai_feature(pre, a.audio + pos, &features[k * kFeatureSize]);

        for (int caminho = 0; caminho < 2; ++caminho) {
            if (caminho == 0) {
                // API do wrapper: float. As features são tratadas como valores
                // reais e requantizadas com o scale do classificador.
                static float f[kFeatureCount * kFeatureSize];
                for (int i = 0; i < kFeatureCount * kFeatureSize; ++i) f[i] = features[i];
                SetTensorArray(in, f, kFeatureCount * kFeatureSize, kFeatureCount * kFeatureSize);
            } else {
                // API crua: os códigos int8 entram como estão, como no exemplo
                // oficial do TFLM.
                CHECK(SetTensorRawInt8(in, features, kFeatureCount * kFeatureSize) == 0,
                      "SetTensorRawInt8 recusou as features");
            }
            if (caminho == 0) {
                // Quanto a requantizacao pela API float alterou a entrada, em
                // relacao aos codigos que o classificador espera.
                int diferentes = 0, saturados = 0;
                for (int i = 0; i < kFeatureCount * kFeatureSize; ++i) {
                    int8_t v = tensor_de(in)->data.int8[i];
                    diferentes += (v != features[i]);
                    saturados += (v == 127 || v == -128);
                }
                printf("  %-4s API float alterou %d de %d bytes da entrada (%d saturados em +-127/-128)\n",
                       a.rotulo, diferentes, kFeatureCount * kFeatureSize, saturados);
            }
            InvokeInterpreter(cls);
            int melhor = 0;
            float p[4];
            for (int i = 0; i < 4; ++i) {
                p[i] = GetTensorAsFloat(out, i);
                if (p[i] > p[melhor]) melhor = i;
            }
            printf("  %-4s via %-12s -> %-7s (silence %.2f unknown %.2f yes %.2f no %.2f)\n",
                   a.rotulo, caminho == 0 ? "API float" : "API crua",
                   kRotulos[melhor], p[0], p[1], p[2], p[3]);
            if (caminho == 1)
                CHECK(strcmp(kRotulos[melhor], a.rotulo) == 0,
                      "audio '%s' classificado como '%s'", a.rotulo, kRotulos[melhor]);
        }
    }
    tflm_free_model(pre);
    tflm_free_model(cls);
}

int main() {
    testa_senoide("test/tflm/models/hello_world_float.tflite");
    testa_senoide("test/tflm/models/hello_world_int8.tflite");
    testa_ciclo_de_vida();
    testa_spam();
    testa_person_detect();
    testa_micro_speech();
    printf("\n%s (%d falha%s)\n", g_falhas ? "FALHOU" : "OK", g_falhas,
           g_falhas == 1 ? "" : "s");
    return g_falhas ? 1 : 0;
}
