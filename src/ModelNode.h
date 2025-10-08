#include "Node.h"
#include <string>

// Forward declarations
class ParamsCall;
class FunctionImpl;

// LLVM Forward declarations
namespace llvm {
    class Value;
    class BasicBlock;
    class GlobalVariable;
}
using namespace llvm;

// Include TFLM wrapper types
extern "C" {
    typedef struct TFLM_Instance TFLM_Instance;
    typedef struct TfLiteTensor TfLiteTensor;
}

class ModelNode : public Node {
public:
    std::string modelName;      // nome do modelo (variável)
    std::string tfliteFile;     // nome do arquivo .tflite extraído
    Node *fileParamNode;        // nó com nome do arquivo .tflite (lazy evaluation)
    Node *arenaSize;            // tamanho da arena
    Node *kernels;              // kernels
    ParamsCall *params;         // parâmetros completos

    // Constructor para declaração de modelo: model nome_modelo("arquivo.tflite", arena_size, kernels)
    ModelNode(const char *name, ParamsCall *p, location_t l);
    
    // Constructor para acesso a membros: nome_modelo.input, nome_modelo.output, etc.
    ModelNode(const char *name, const char *member, location_t l);
    
    // Constructor para atribuições: nome_modelo.input = valor
    ModelNode(const char *name, const char *member, Node *value, location_t l);

    virtual Value* generate(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) override;
    virtual Value* generateDeclaration(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock);
    virtual Value* generateModelInitialization(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock, 
                                               GlobalVariable* modelDataGlobal, GlobalVariable* modelLenGlobal);
    virtual Value* generateMemberAccess(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock);
    
private:
    std::string memberName;     // para acesso a membros
    Node *assignedValue;        // valor atribuído
};

