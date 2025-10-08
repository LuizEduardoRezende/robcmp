#include "ModelNode.h"
#include "ParamsCall.h"
#include "StringConst.h"
#include "BackLLVM.h"
#include "FunctionImpl.h"
#include "../wrappers/tflm/tflm_wrapper.h"
#include <iostream>
#include <fstream>
#include <vector>

// Constructor para declaração de modelo: model nome_modelo("arquivo.tflite", arena_size, kernels)
ModelNode::ModelNode(const char *name, ParamsCall *p, location_t l) 
: Node(l), modelName(name), params(p), arenaSize(nullptr), kernels(nullptr), assignedValue(nullptr), fileParamNode(nullptr) {
    
    if (params && params->getNumParams() >= 3) {

        Node *fileParam = params->getParamElement(0);
        fileParamNode = fileParam;
        
        StringConst *strConst = dynamic_cast<StringConst*>(fileParam);
        if (strConst) {
            // Usar método getter público (precisa ser adicionado em StringConst.h)
            tfliteFile = strConst->getStringValue();
        }
        
        // Segundo parâmetro: arena size
        arenaSize = params->getParamElement(1);
        
        // Terceiro parâmetro: kernels
        kernels = params->getParamElement(2);
    }
}

// Constructor para acesso a membros
ModelNode::ModelNode(const char *name, const char *member, location_t l)
    : Node(l), modelName(name), memberName(member), fileParamNode(nullptr), params(nullptr), arenaSize(nullptr), kernels(nullptr), assignedValue(nullptr) {
}

// Constructor para atribuições
ModelNode::ModelNode(const char *name, const char *member, Node *value, location_t l)
    : Node(l), modelName(name), memberName(member), fileParamNode(nullptr), assignedValue(value), params(nullptr), arenaSize(nullptr), kernels(nullptr) {
}

Value* ModelNode::generate(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) {
    if (params) {
        // Declaração do modelo - gerar dados estáticos e inicialização
        return generateDeclaration(func, block, allocblock);
    } else if (!memberName.empty()) {
        // Acesso a membros (modelo.input, modelo.output, etc.)
        return generateMemberAccess(func, block, allocblock);
    }
    return nullptr;
}

Value* ModelNode::generateDeclaration(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) {
    // Inicializar escopo se necessário
    if (!getScope()) {
        setScope(func);
    }
    
    // Definir escopo nos parâmetros também
    if (arenaSize && !arenaSize->getScope()) {
        arenaSize->setScope(func);
    }
    if (kernels && !kernels->getScope()) {
        kernels->setScope(func);
    }
    if (fileParamNode && !fileParamNode->getScope()) {
        fileParamNode->setScope(func);
    }
    
    // Verificar se conseguimos extrair o nome do arquivo
    if (tfliteFile.empty()) {
        yyerrorcpp("Nome do arquivo .tflite não foi extraído corretamente para o modelo '" + modelName + "'.", this);
        setSemanticError();
        return nullptr;
    }
    
    // Construir caminho relativo ao arquivo .rob usando source_filename do módulo
    std::string sourceFile = mainmodule->getSourceFileName();
    std::string sourceDir = "./"; // default
    
    size_t lastSlash = sourceFile.find_last_of('/');
    if (lastSlash != std::string::npos) {
        sourceDir = sourceFile.substr(0, lastSlash + 1);
    }
    
    std::string fullPath = sourceDir + tfliteFile;
    std::ifstream file(fullPath);
    if (!file.good()) {
        yyerrorcpp("Arquivo '" + fullPath + "' não encontrado para o modelo '" + modelName + "'.", this);
        setSemanticError();
        return nullptr;
    }
    
    tfliteFile = fullPath;
    
    // Ler arquivo
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<unsigned char> buffer(fileSize);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();
    
    // Arquivo carregado com sucesso
    
    // Criar array global com os dados do modelo usando LLVM
    Type* i8Type = Type::getInt8Ty(global_context);
    ArrayType* modelDataType = ArrayType::get(i8Type, fileSize);
    
    // Converter buffer para array de constantes LLVM
    std::vector<Constant*> modelDataConstants;
    modelDataConstants.reserve(fileSize);
    for (size_t i = 0; i < fileSize; i++) {
        modelDataConstants.push_back(ConstantInt::get(i8Type, buffer[i]));
    }
    
    // Criar array global constante com os dados do modelo
    std::string modelDataName = modelName + "_model_data";
    GlobalVariable* modelDataGlobal = new GlobalVariable(
        *mainmodule, 
        modelDataType,
        true,  // é constante
        GlobalValue::InternalLinkage,
        ConstantArray::get(modelDataType, modelDataConstants),
        modelDataName
    );
    
    // Criar global para o tamanho do modelo
    std::string modelLenName = modelName + "_model_len";
    Type* i32Type = Type::getInt32Ty(global_context);
    GlobalVariable* modelLenGlobal = new GlobalVariable(
        *mainmodule,
        i32Type,
        true,  // é constante
        GlobalValue::InternalLinkage,
        ConstantInt::get(i32Type, fileSize),
        modelLenName
    );
    
    // Gerar inicialização do interpretador
    return generateModelInitialization(func, block, allocblock, modelDataGlobal, modelLenGlobal);
}



Value* ModelNode::generateModelInitialization(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock, 
                                            GlobalVariable* modelDataGlobal, GlobalVariable* modelLenGlobal) {
    
    Builder->SetInsertPoint(allocblock);
    
    // Criar arena estática (aloca uma única vez)
    Value* arenaSizeValue;
    if (arenaSize) {
        arenaSizeValue = arenaSize->generate(func, block, allocblock);
    } else {
        arenaSizeValue = ConstantInt::get(Type::getInt32Ty(global_context), 2048);
    }
    
    // Converter para ConstantInt se possível para criar array estático
    ConstantInt* constantArenaSize = dyn_cast<ConstantInt>(arenaSizeValue);
    if (!constantArenaSize) {
        yyerrorcpp("Tamanho da arena do modelo '" + modelName + "' deve ser uma constante.", this);
        setSemanticError();
        return nullptr;
    }
    
    uint64_t arenaSize = constantArenaSize->getZExtValue();
    Type* i8Type = Type::getInt8Ty(global_context);
    ArrayType* arenaType = ArrayType::get(i8Type, arenaSize);
    
    // Criar arena global estática
    std::string arenaName = modelName + "_arena";
    GlobalVariable* arenaGlobal = new GlobalVariable(
        *mainmodule,
        arenaType,
        false, // não é constante (será modificada)
        GlobalValue::InternalLinkage,
        ConstantAggregateZero::get(arenaType), // inicializar com zeros
        arenaName
    );
    
    // Criar variável global para a instância do interpretador
    std::string instanceName = modelName + "_instance";
    PointerType* ptrType = PointerType::getUnqual(Type::getInt8Ty(global_context));
    GlobalVariable* instanceGlobal = new GlobalVariable(
        *mainmodule,
        ptrType,
        false, // não é constante
        GlobalValue::InternalLinkage,
        ConstantPointerNull::get(ptrType),
        instanceName
    );
    
    Builder->SetInsertPoint(block);
    
    // Verificar se já foi inicializado (lazy initialization)
    Value* currentInstance = Builder->CreateLoad(ptrType, instanceGlobal, "current_instance");
    Value* isNull = Builder->CreateICmpEQ(currentInstance, 
        ConstantPointerNull::get(ptrType), 
        "is_null");
    
    // Criar blocos para inicialização condicional
    BasicBlock* initBlock = BasicBlock::Create(global_context, "init_model", func->getLLVMFunction());
    BasicBlock* afterInitBlock = BasicBlock::Create(global_context, "after_init", func->getLLVMFunction());
    
    Builder->CreateCondBr(isNull, initBlock, afterInitBlock);
    
    // Bloco de inicialização
    Builder->SetInsertPoint(initBlock);
    
    // Preparar argumentos para InitializeInterpreterAuto
    PointerType* i8PtrType = PointerType::getUnqual(Type::getInt8Ty(global_context));
    Value* modelDataPtr = Builder->CreateBitCast(modelDataGlobal, i8PtrType);
    Value* dummyInt = ConstantInt::get(Type::getInt32Ty(global_context), 0);
    Value* arenaPtr = Builder->CreateBitCast(arenaGlobal, i8PtrType);
    Value* arenaSizeArg = ConstantInt::get(Type::getInt64Ty(global_context), arenaSize);
    
    // Criar call para InitializeInterpreterAuto
    std::vector<Type*> argTypes = {
        i8PtrType, // model_data
        Type::getInt32Ty(global_context),   // dummy int
        i8PtrType, // arena
        Type::getInt32Ty(global_context),   // dummy int  
        Type::getInt64Ty(global_context)    // arena_size
    };
    
    FunctionType* funcType = FunctionType::get(i8PtrType, argTypes, false);
    FunctionCallee initFunc = mainmodule->getOrInsertFunction("InitializeInterpreterAuto", funcType);
    
    std::vector<Value*> args = {modelDataPtr, dummyInt, arenaPtr, dummyInt, arenaSizeArg};
    Value* newInstance = Builder->CreateCall(initFunc, args, "new_instance");
    
    // Armazenar nova instância
    Builder->CreateStore(newInstance, instanceGlobal);
    
    // TODO: Adicionar verificação de erro (se newInstance == NULL)
    
    Builder->CreateBr(afterInitBlock);
    
    // Continuar após inicialização
    Builder->SetInsertPoint(afterInitBlock);
    
    // Retornar a instância (carregada novamente para garantir valor atual)
    return Builder->CreateLoad(ptrType, instanceGlobal, "model_instance");
}

Value* ModelNode::generateMemberAccess(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) {
    // Implementar acesso a membros do modelo (input, output, etc.)
    // Por enquanto, retornar nullptr - implementar conforme necessário
    std::string memberVariableName = modelName + "_" + memberName;
    
    if (assignedValue) {
        // Atribuição: modelo.input = valor
        // TODO: Implementar lógica de atribuição
        Value* value = assignedValue->generate(func, block, allocblock);
        return value;
    } else {
        // Acesso: modelo.input
        // TODO: Implementar lógica de acesso a membro
        // Por enquanto retornar nullptr
        return nullptr;
    }
}



