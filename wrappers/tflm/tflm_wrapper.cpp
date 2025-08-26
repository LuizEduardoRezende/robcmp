#include "third-party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "third-party/tflite-micro/tensorflow/lite/schema/schema_generated.h"
#include "third-party/tflite-micro/tensorflow/lite/c/common.h"
#include "third-party/tflite-micro/tensorflow/lite/micro/micro_log.h"
#include <cstring>

extern "C" {

using MutableResolver = tflite::MicroMutableOpResolver<10>;

struct TFLM_Instance {
    tflite::MicroInterpreter* interpreter;
    MutableResolver* resolver;
};

void* InitializeInterpreter(const uint8_t* model_data, uint8_t* tensor_arena, size_t tensor_arena_size) {
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema version mismatch!");
        return nullptr;
    }

    TFLM_Instance* instance = new TFLM_Instance();
    

    instance->resolver = new MutableResolver();

    instance->resolver->AddConv2D();
    instance->resolver->AddMaxPool2D();
    instance->resolver->AddFullyConnected();
    instance->resolver->AddReshape();
    instance->resolver->AddSoftmax();
    instance->resolver->AddQuantize();
    instance->resolver->AddGather(); 
    // Adicione outros kernels conforme necessário

    
    instance->interpreter = new tflite::MicroInterpreter(model, *(instance->resolver), tensor_arena, tensor_arena_size);

    if (instance->interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("Failed to allocate tensors! Verifique se todas as ops foram registradas.");
        delete instance->interpreter;
        delete instance->resolver;
        delete instance;
        return nullptr;
    }

    MicroPrintf("Interpreter initialized successfully.");
    return reinterpret_cast<void*>(instance);
}

void DestroyInterpreter(void* instance_handle) {
    if (instance_handle == nullptr) return;

    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    delete instance->interpreter;
    delete instance->resolver; 
    delete instance;
    MicroPrintf("Interpreter destroyed successfully.");
}

TfLiteTensor* GetInputTensor(void* instance_handle, size_t index) {
    if (instance_handle == nullptr) return nullptr;
    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    return instance->interpreter->input(index);
}

const TfLiteTensor* GetOutputTensor(void* instance_handle, size_t index) {
    if (instance_handle == nullptr) return nullptr;
    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    return instance->interpreter->output(index);
}

void InvokeInterpreter(void* instance_handle) {
    if (instance_handle == nullptr) {
        MicroPrintf("Interpreter pointer is null.");
        return;
    }

    TFLM_Instance* instance = reinterpret_cast<TFLM_Instance*>(instance_handle);
    if (instance->interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Failed to invoke interpreter.");
    } else {
        MicroPrintf("Interpreter invoked successfully.");
    }
}

void RegisterOp(tflite::MicroMutableOpResolver<10>* resolver, tflite::BuiltinOperator op) {
    switch (op) {
        case tflite::BuiltinOperator_CONV_2D:
            resolver->AddConv2D();
            break;
        case tflite::BuiltinOperator_MAX_POOL_2D:
            resolver->AddMaxPool2D();
            break;
        case tflite::BuiltinOperator_FULLY_CONNECTED:
            resolver->AddFullyConnected();
            break;
        case tflite::BuiltinOperator_RESHAPE:
            resolver->AddReshape();
            break;
        case tflite::BuiltinOperator_SOFTMAX:
            resolver->AddSoftmax();
            break;
        // Adicione outros casos para as operações que você quer suportar
        default:
            MicroPrintf("Operação não suportada no benchmark: %d", op);
            break;
    }
}

int RunBenchmark(const uint8_t* model_data, uint8_t* tensor_arena, size_t tensor_arena_size, int num_invocations) {
    MicroPrintf("--- Iniciando Análise de Benchmark ---");

    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Erro: Schema do modelo incompatível!");
        return -1;
    }

    // Pega o primeiro (e geralmente único) grafo do modelo
    const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
    // Pega a lista de todos os tipos de operadores usados no modelo
    const auto* opcodes = model->operator_codes();

    // --- Etapa 1: Descobrir os Kernels Necessários ---
    MicroPrintf("Kernels necessários para este modelo:");
    // Usamos um truque para evitar imprimir duplicatas.
    // Este array de booleanos marcará os opcodes que já imprimimos.
    bool printed_opcodes[opcodes->size()];
    for(size_t i = 0; i < opcodes->size(); ++i) { printed_opcodes[i] = false; }

    for (size_t i = 0; i < subgraph->operators()->size(); ++i) {
        const tflite::Operator* op = subgraph->operators()->Get(i);
        const uint32_t opcode_index = op->opcode_index();
        if (!printed_opcodes[opcode_index]) {
            const tflite::OperatorCode* opcode = opcodes->Get(opcode_index);
            const tflite::BuiltinOperator builtin_code = opcode->builtin_code(); // ← Mudança aqui
            MicroPrintf("  - %s", tflite::EnumNameBuiltinOperator(builtin_code));
            printed_opcodes[opcode_index] = true;
        }
    }

    // --- Etapa 2: Registrar os Kernels e Executar o Benchmark ---
    tflite::MicroMutableOpResolver<10> resolver;

    // Agora registramos os kernels que descobrimos
    for (size_t i = 0; i < subgraph->operators()->size(); ++i) {
        const tflite::Operator* op = subgraph->operators()->Get(i);
        const tflite::OperatorCode* opcode = opcodes->Get(op->opcode_index());
        const tflite::BuiltinOperator builtin_code = opcode->builtin_code();
        RegisterOp(&resolver, builtin_code);
    }
    
    // Configura o interpretador
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, tensor_arena_size);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        MicroPrintf("Erro: Falha ao alocar tensores.");
        return -2;
    }

    MicroPrintf("\nIniciando invocações de benchmark...");

    for (int i = 0; i < num_invocations; ++i) {
        if (interpreter.Invoke() != kTfLiteOk) {
            MicroPrintf("Erro: Falha ao invocar o interpretador.");
            return -3;
        }
        MicroPrintf("  - Invocação %d concluída.", i + 1);
    }
        
    MicroPrintf("Benchmark concluído com sucesso.");
    MicroPrintf("--- Fim da Análise ---");
    return 0; // Sucesso
}

} // extern "C"