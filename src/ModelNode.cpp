#include "ModelNode.h"
#include "BackLLVM.h"
#include "FunctionImpl.h"
#include "../wrappers/tflm/tflm_wrapper.h"

#include "ParamsCall.h"
#include "StringConst.h"
#include "Load.h"

#include "Array.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

ModelNode::ModelNode(const char *name, ParamsCall *p, location_t l) 
: Node(l), modelName(name), params(p), arenaSize(nullptr), kernels(nullptr), assignedValue(nullptr), tensorIndex(nullptr), fileParamNode(nullptr) {
    
    if (params && params->getNumParams() >= 3) {
        Node *fileParam = params->getParamElement(0);
        fileParamNode = fileParam;
        
        StringConst *strConst = dynamic_cast<StringConst*>(fileParam);
        if (strConst) {
            tfliteFile = strConst->getStringValue();
        }
        
        arenaSize = params->getParamElement(1);
        kernels = params->getParamElement(2);
    }
}

ModelNode::ModelNode(const char *name, const char *member, location_t l)
    : Node(l), modelName(name), memberName(member), fileParamNode(nullptr), params(nullptr), arenaSize(nullptr), kernels(nullptr), assignedValue(nullptr), tensorIndex(nullptr) {
}

ModelNode::ModelNode(const char *name, const char *member, Node *value, location_t l)
    : Node(l), modelName(name), memberName(member), fileParamNode(nullptr), assignedValue(value), tensorIndex(nullptr), params(nullptr), arenaSize(nullptr), kernels(nullptr) {
}

ModelNode::ModelNode(const char *name, const char *member, Node *index, Node *value, location_t l)
    : Node(l), modelName(name), memberName(member), fileParamNode(nullptr), assignedValue(value), tensorIndex(index), params(nullptr), arenaSize(nullptr), kernels(nullptr) {
}

Value* ModelNode::generate(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) {
    if (!getScope()) {
        setScope(func);
    }
    
    if (params) {
        return generateDeclaration(func, block, allocblock);
    } else if (!memberName.empty()) {
        return generateMemberAccess(func, block, allocblock);
    }
    return nullptr;
}

Value* ModelNode::generateDeclaration(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) {
    if (!getScope()) {
        setScope(func);
    }
    
    if (params) {
        for (int i = 0; i < params->getNumParams(); i++) {
            Node* param = params->getParamElement(i);
            if (param && !param->getScope()) {
                param->setScope(func);
            }
        }
    }
    
    if (tfliteFile.empty()) {
        yyerrorcpp("Nome do arquivo .tflite não foi extraído corretamente para o modelo '" + modelName + "'.", this);
        setSemanticError();
        return nullptr;
    }
    
    // Construir caminho relativo ao arquivo .rob
    std::string sourceFile = mainmodule->getSourceFileName();
    std::string sourceDir = "./";
    
    size_t lastSlash = sourceFile.find_last_of('/');
    if (lastSlash != std::string::npos) {
        sourceDir = sourceFile.substr(0, lastSlash + 1);
    }
    
    std::string fullPath = sourceDir + tfliteFile;
    
    // Abrir arquivo usando POSIX
    int fd = open(fullPath.c_str(), O_RDONLY);
    if (fd == -1) {
        yyerrorcpp("Arquivo '" + fullPath + "' não encontrado para o modelo '" + modelName + "'.", this);
        setSemanticError();
        return nullptr;
    }
    
    // Obter tamanho do arquivo
    struct stat st;
    if (fstat(fd, &st) == -1) {
        close(fd);
        yyerrorcpp("Erro ao obter tamanho do arquivo '" + fullPath + "'.", this);
        setSemanticError();
        return nullptr;
    }
    
    tfliteFile = fullPath;
    size_t fileSize = st.st_size;
    
    // Ler arquivo .tflite
    std::vector<unsigned char> buffer(fileSize);
    ssize_t bytesRead = read(fd, buffer.data(), fileSize);
    close(fd);
    
    if (bytesRead != static_cast<ssize_t>(fileSize)) {
        yyerrorcpp("Erro ao ler arquivo '" + fullPath + "' completamente.", this);
        setSemanticError();
        return nullptr;
    }
    
    // Criar array global com os dados do modelo
    Type* i8Type = Type::getInt8Ty(global_context);
    ArrayType* modelDataType = ArrayType::get(i8Type, fileSize);
    
    std::vector<Constant*> modelDataConstants;
    modelDataConstants.reserve(fileSize);
    for (size_t i = 0; i < fileSize; i++) {
        modelDataConstants.push_back(ConstantInt::get(i8Type, buffer[i]));
    }
    
    std::string modelDataName = modelName + "_model_data";
    GlobalVariable* modelDataGlobal = new GlobalVariable(
        *mainmodule, 
        modelDataType,
        true,
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
        true,
        GlobalValue::InternalLinkage,
        ConstantInt::get(i32Type, fileSize),
        modelLenName
    );

    // VALIDAÇÕES SEMÂNTICAS PRIMEIRO (antes de qualquer geração de código LLVM)
    if (!kernels) {
        yyerrorcpp("Erro: Modelo '" + modelName + "' deve especificar um array de kernels necessários.", this);
        setSemanticError();
        return nullptr;
    }
    
    // Detectar o tamanho do array de kernels ANTES de gerar código LLVM
    Node* kernelsSymbol = nullptr;
    size_t arraySize = 0;
    
    if (Load* loadNode = dynamic_cast<Load*>(kernels)) {
        kernelsSymbol = loadNode->getIdentSymbol(false);
    }
    
    if (kernelsSymbol) {
        if (Array* arrayNode = dynamic_cast<Array*>(kernelsSymbol)) {
            arraySize = arrayNode->getSize();
        } else {
            arraySize = 1;
        }
    } else {
        yyerrorcpp("Erro: Não foi possível determinar o símbolo do array de kernels para o modelo '" + modelName + "'.", this);
        setSemanticError();
        return nullptr;
    }
    
    if (arraySize == 0) {
        yyerrorcpp("Erro: Array de kernels do modelo '" + modelName + "' não pode estar vazio.", this);
        setSemanticError();
        return nullptr;
    }
    
    if (arraySize > 255) {
        yyerrorcpp("Erro: Array de kernels do modelo '" + modelName + "' tem " + std::to_string(arraySize) + " elementos, mas o máximo suportado é 255.", this);
        setSemanticError();
        return nullptr;
    }

    Builder->SetInsertPoint(allocblock);
    
    Value* arenaSizeValue;
    if (arenaSize) {
        arenaSizeValue = arenaSize->generate(func, block, allocblock);
    } else {
        arenaSizeValue = ConstantInt::get(Type::getInt32Ty(global_context), 2048);
    }
    
    ConstantInt* constantArenaSize = dyn_cast<ConstantInt>(arenaSizeValue);
    if (!constantArenaSize) {
        yyerrorcpp("Tamanho da arena do modelo '" + modelName + "' deve ser uma constante.", this);
        setSemanticError();
        return nullptr;
    }
    
    uint64_t arenaSize = constantArenaSize->getZExtValue();
    ArrayType* arenaType = ArrayType::get(i8Type, arenaSize);
    
    // Criar arena global
    std::string arenaName = modelName + "_arena";
    GlobalVariable* arenaGlobal = new GlobalVariable(
        *mainmodule,
        arenaType,
        false,
        GlobalValue::InternalLinkage,
        ConstantAggregateZero::get(arenaType),
        arenaName
    );
    
    // Criar variável global para instância do interpretador
    std::string instanceName = modelName + "_instance";
    PointerType* ptrType = PointerType::getUnqual(Type::getInt8Ty(global_context));
    GlobalVariable* instanceGlobal = new GlobalVariable(
        *mainmodule,
        ptrType,
        false,
        GlobalValue::InternalLinkage,
        ConstantPointerNull::get(ptrType),
        instanceName
    );
    
    Builder->SetInsertPoint(block);
    
    // Implementar lazy initialization
    Value* currentInstance = Builder->CreateLoad(ptrType, instanceGlobal, "current_instance");
    Value* isNull = Builder->CreateICmpEQ(currentInstance, 
        ConstantPointerNull::get(ptrType), 
        "is_null");
    
    BasicBlock* initBlock = BasicBlock::Create(global_context, "init_model", func->getLLVMFunction());
    BasicBlock* afterInitBlock = BasicBlock::Create(global_context, "after_init", func->getLLVMFunction());
    
    Builder->CreateCondBr(isNull, initBlock, afterInitBlock);
    
    Builder->SetInsertPoint(initBlock);
    
    // Preparar argumentos para InitializeInterpreter
    PointerType* i8PtrType = PointerType::getUnqual(Type::getInt8Ty(global_context));
    Value* modelDataPtr = Builder->CreateBitCast(modelDataGlobal, i8PtrType);
    Value* arenaPtr = Builder->CreateBitCast(arenaGlobal, i8PtrType);
    
    // Gerar código LLVM para o array de kernels (validações já foram feitas)
    Value* kernelsArrayValue = kernels->generate(func, initBlock, allocblock);
    if (!kernelsArrayValue) {
        // ERRO: Criar terminador antes de retornar
        Builder->CreateStore(ConstantPointerNull::get(ptrType), instanceGlobal);
        Builder->CreateBr(afterInitBlock);
        yyerrorcpp("Erro: Não foi possível gerar código para o array de kernels do modelo '" + modelName + "'.", this);
        setSemanticError();
        return nullptr;
    }
    
    Value* kernelsArrayPtr = Builder->CreateBitCast(kernelsArrayValue, i8PtrType);
    Value* numKernels = ConstantInt::get(Type::getInt8Ty(global_context), (uint8_t)arraySize);
    
    // Continuar com a geração dos argumentos para InitializeInterpreter
    Value* dummyInt = ConstantInt::get(Type::getInt32Ty(global_context), 0);
    Value* arenaSizeArg = ConstantInt::get(Type::getInt32Ty(global_context), (uint32_t)arenaSize);
    
    std::vector<Type*> argTypes = {
        i8PtrType,                          // model_data
        i8PtrType,                          // tensor_arena  
        i8PtrType,                          // required_kernels
        Type::getInt32Ty(global_context),   // dummy int
        Type::getInt32Ty(global_context),   // tensor_arena_size
        Type::getInt8Ty(global_context)     // num_kernels
    };
    
    FunctionType* funcType = FunctionType::get(Type::getInt64Ty(global_context), argTypes, false);
    FunctionCallee initFunc = mainmodule->getOrInsertFunction("InitializeInterpreter", funcType);
    
    std::vector<Value*> args = {modelDataPtr, arenaPtr, kernelsArrayPtr, dummyInt, arenaSizeArg, numKernels};
    Value* newInstance = Builder->CreateCall(initFunc, args, "new_instance");
    
    // Converter uintptr_t para ponteiro
    Value* instancePtr = Builder->CreateIntToPtr(newInstance, ptrType, "instance_ptr");
    Builder->CreateStore(instancePtr, instanceGlobal);
    Builder->CreateBr(afterInitBlock);
    
    Builder->SetInsertPoint(afterInitBlock);
    
    return Builder->CreateLoad(ptrType, instanceGlobal, "model_instance");
}

Value* ModelNode::generateMemberAccess(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) {
    if (!getScope()) {
        setScope(func);
    }
    
    if (assignedValue && !assignedValue->getScope()) {
        assignedValue->setScope(func);
    }
    
    Builder->SetInsertPoint(block);
    
    std::string instanceName = modelName + "_instance";
    GlobalVariable* instanceGlobal = mainmodule->getNamedGlobal(instanceName);
    if (!instanceGlobal) {
        yyerrorcpp("Modelo '" + modelName + "' não foi declarado antes do acesso ao membro '" + memberName + "'.", this);
        setSemanticError();
        return nullptr;
    }
    
    PointerType* ptrType = PointerType::getUnqual(Type::getInt8Ty(global_context));
    Value* modelInstance = Builder->CreateLoad(ptrType, instanceGlobal, "model_instance");
    
    if (memberName == "input") {
        return generateInputAccess(func, block, allocblock, modelInstance);
    } else if (memberName == "output") {
        return generateOutputAccess(func, block, allocblock, modelInstance);
    } else if (memberName == "invoke") {
        return generateInvoke(func, block, allocblock, modelInstance);
    } else {
        yyerrorcpp("Membro '" + memberName + "' não suportado no modelo '" + modelName + "'. Use 'input', 'output' ou 'invoke'.", this);
        setSemanticError();
        return nullptr;
    }
}

Value* ModelNode::generateInputAccess(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock, Value* modelInstance) {
    if(!assignedValue){
        yyerrorcpp("Não é possível ler diretamente do tensor de entrada '" + modelName + ".input'. Use apenas para atribuição: '" + modelName + ".input = dados'.", this);
        setSemanticError();
        return nullptr;
    }  

    PointerType* i8PtrType = PointerType::getUnqual(Type::getInt8Ty(global_context));
    
    // Use o índice fornecido ou 0 como padrão
    Value* tensorIndexValue;
    if (tensorIndex) {
        tensorIndexValue = tensorIndex->generate(func, block, allocblock);
        if (!tensorIndexValue) {
            yyerrorcpp("Erro ao gerar índice do tensor de entrada.", this);
            setSemanticError();
            return nullptr;
        }
        // Converter para int64 se necessário
        if (tensorIndexValue->getType()->isIntegerTy() && !tensorIndexValue->getType()->isIntegerTy(64)) {
            tensorIndexValue = Builder->CreateZExt(tensorIndexValue, Type::getInt64Ty(global_context), "tensor_index_ext");
        }
    } else {
        tensorIndexValue = ConstantInt::get(Type::getInt64Ty(global_context), 0);
    }
    
    Value* dummyInt = ConstantInt::get(Type::getInt32Ty(global_context), 0);
    
    FunctionType* getInputType = FunctionType::get(
        Type::getInt64Ty(global_context),
        {Type::getInt64Ty(global_context), Type::getInt64Ty(global_context), Type::getInt32Ty(global_context)},
        false
    );
    FunctionCallee getInputFunc = mainmodule->getOrInsertFunction("GetInputTensor", getInputType);
    
    Value* modelHandle = Builder->CreatePtrToInt(modelInstance, Type::getInt64Ty(global_context));
    Value* tensorHandle = Builder->CreateCall(getInputFunc, {modelHandle, tensorIndexValue, dummyInt}, "input_tensor");
    
    FunctionType* getSizeType = FunctionType::get(
        Type::getInt64Ty(global_context),
        {Type::getInt64Ty(global_context), Type::getInt32Ty(global_context)},
        false
    );
    FunctionCallee getSizeFunc = mainmodule->getOrInsertFunction("GetTensorSize", getSizeType);
    Value* tensorSize = Builder->CreateCall(getSizeFunc, {tensorHandle, dummyInt}, "tensor_size");
    
    Load* loadNode = dynamic_cast<Load*>(assignedValue);
    if (loadNode) {
        // Para Load nodes, sempre tentar como array primeiro
        return generateVariableArrayToTensorCopy(func, block, allocblock, tensorHandle, tensorSize, loadNode);
    } else {
        Value* assignedData = assignedValue->generate(func, block, allocblock);
        if (!assignedData) {
            yyerrorcpp("Erro ao gerar dados para atribuição ao tensor de entrada.", this);
            setSemanticError();
            return nullptr;
        }
        return generateScalarToTensorCopy(func, block, allocblock, tensorHandle, assignedData);
    }
}

Value* ModelNode::generateOutputAccess(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock, Value* modelInstance) {
    // Use o índice fornecido ou 0 como padrão (índice do tensor de saída, não índice dentro do tensor)
    Value* tensorIndexValue;
    if (tensorIndex) {
        tensorIndexValue = tensorIndex->generate(func, block, allocblock);
        if (!tensorIndexValue) {
            yyerrorcpp("Erro ao gerar índice do tensor de saída.", this);
            setSemanticError();
            return nullptr;
        }
        // Converter para int64 se necessário
        if (tensorIndexValue->getType()->isIntegerTy() && !tensorIndexValue->getType()->isIntegerTy(64)) {
            tensorIndexValue = Builder->CreateZExt(tensorIndexValue, Type::getInt64Ty(global_context), "tensor_index_ext");
        }
    } else {
        tensorIndexValue = ConstantInt::get(Type::getInt64Ty(global_context), 0);
    }
    
    Value* dummyInt = ConstantInt::get(Type::getInt32Ty(global_context), 0);
    
    // Obter o handle do tensor de saída
    FunctionType* getOutputType = FunctionType::get(
        Type::getInt64Ty(global_context),
        {Type::getInt64Ty(global_context), Type::getInt64Ty(global_context), Type::getInt32Ty(global_context)},
        false
    );
    FunctionCallee getOutputFunc = mainmodule->getOrInsertFunction("GetOutputTensor", getOutputType);
    
    Value* modelHandle = Builder->CreatePtrToInt(modelInstance, Type::getInt64Ty(global_context));
    Value* tensorHandle = Builder->CreateCall(getOutputFunc, {modelHandle, tensorIndexValue, dummyInt}, "output_tensor");

    // Chamar AllocAndGetTensorArray
    PointerType* floatPtrType = PointerType::getUnqual(Type::getFloatTy(global_context));
    PointerType* sizeTPtrType = PointerType::getUnqual(Type::getInt64Ty(global_context));
    FunctionType* allocGetArrayType = FunctionType::get(
        floatPtrType,
        {Type::getInt64Ty(global_context), sizeTPtrType, Type::getInt32Ty(global_context)},
        false
    );
    FunctionCallee allocGetArrayFunc = mainmodule->getOrInsertFunction("AllocAndGetTensorArray", allocGetArrayType);

    // Alocar variável temporária para size_t (opcional)
    Value* outSizePtr = nullptr;
    outSizePtr = ConstantPointerNull::get(sizeTPtrType);
    Value* floatArray = Builder->CreateCall(allocGetArrayFunc, {tensorHandle, outSizePtr, dummyInt}, "float_array");
    // DEBUG: Print first value of floatArray
    {
        FunctionType* printfType = FunctionType::get(IntegerType::getInt32Ty(global_context), PointerType::get(Type::getInt8Ty(global_context), 0), true);
        FunctionCallee printfFunc = mainmodule->getOrInsertFunction("printf", printfType);
        Value* fmtStr = Builder->CreateGlobalString("[DEBUG] ModelNode floatArray[0]: %f\n");
        Value* firstValue = Builder->CreateLoad(Type::getFloatTy(global_context), floatArray, "first_output_value");
        Builder->CreateCall(printfFunc, {fmtStr, firstValue});
    }
    return floatArray;
}

Value* ModelNode::generateVariableArrayToTensorCopy(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock, 
                                                   Value* tensorHandle, Value* tensorSize, Load* loadNode) {
    // Detectar tamanho do array através do símbolo original
    size_t arraySize = 1;
    Node* originalSymbol = loadNode->getIdentSymbol(false);
    if (originalSymbol) {
        Array* arrayNode = dynamic_cast<Array*>(originalSymbol);
        if (arrayNode) {
            arraySize = arrayNode->getSize();
        }
    }

    Value* varPtr = assignedValue->generate(func, block, allocblock);
    if (!varPtr) {
        yyerrorcpp("Erro ao obter ponteiro da variável array para cópia ao tensor.", this);
        setSemanticError();
        return tensorHandle;
    }
    
    // VERIFICAÇÃO SEMÂNTICA: Arrays devem ser float*
    PointerType* floatPtrType = PointerType::getUnqual(Type::getFloatTy(global_context));
    if (varPtr->getType() != floatPtrType) {
        yyerrorcpp("ERRO SEMÂNTICO: Array deve conter valores float. Conversões serão feitas no wrapper.", this);
        setSemanticError();
        return tensorHandle;
    }
    
    // Chamar SetTensorArray - wrapper fará todas as conversões necessárias
    Value* dummyInt = ConstantInt::get(Type::getInt32Ty(global_context), 0);
    Value* arrayCount = ConstantInt::get(Type::getInt64Ty(global_context), arraySize);
    
    FunctionType* setArrayType = FunctionType::get(
        Type::getVoidTy(global_context),
        {Type::getInt64Ty(global_context), floatPtrType, Type::getInt64Ty(global_context), Type::getInt32Ty(global_context)},
        false
    );
    FunctionCallee setArrayFunc = mainmodule->getOrInsertFunction("SetTensorArray", setArrayType);
    
    Builder->CreateCall(setArrayFunc, {tensorHandle, varPtr, arrayCount, dummyInt});
    
    return tensorHandle;
}

Value* ModelNode::generateScalarToTensorCopy(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock, Value* tensorHandle, Value* data) {
    // VERIFICAÇÃO SEMÂNTICA: Valores devem ser float
    Value* floatValue;
    if (data->getType()->isPointerTy()) {
        // Ponteiro para float
        if (data->getType() != PointerType::getUnqual(Type::getFloatTy(global_context))) {
            yyerrorcpp("ERRO SEMÂNTICO: Valor deve ser float. Conversões serão feitas no wrapper.", this);
            setSemanticError();
            return tensorHandle;
        }
        floatValue = Builder->CreateLoad(Type::getFloatTy(global_context), data, "loaded_float");
    } else {
        // Valor direto
        if (!data->getType()->isFloatTy()) {
            yyerrorcpp("ERRO SEMÂNTICO: Valor deve ser float. Conversões serão feitas no wrapper.", this);
            setSemanticError();
            return tensorHandle;
        }
        floatValue = data;
    }
    
    // Chamar SetTensorValue - wrapper fará conversões necessárias
    FunctionType* setValueType = FunctionType::get(
        Type::getVoidTy(global_context),
        {Type::getInt64Ty(global_context), Type::getInt64Ty(global_context), Type::getFloatTy(global_context), Type::getInt32Ty(global_context)},
        false
    );
    FunctionCallee setValueFunc = mainmodule->getOrInsertFunction("SetTensorValue", setValueType);
    
    Value* index = ConstantInt::get(Type::getInt64Ty(global_context), 0);
    Value* dummyInt = ConstantInt::get(Type::getInt32Ty(global_context), 0);
    Builder->CreateCall(setValueFunc, {tensorHandle, index, floatValue, dummyInt});
    
    return tensorHandle;
}

Value* ModelNode::generateInvoke(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock, Value* modelInstance) {
    if (assignedValue) {
        yyerrorcpp("Não é possível atribuir valores à operação '" + modelName + ".invoke'.", this);
        setSemanticError();
        return nullptr;
    }
    
    Value* dummyInt = ConstantInt::get(Type::getInt32Ty(global_context), 0);
    
    // Criar chamada para InvokeInterpreter
    FunctionType* invokeType = FunctionType::get(
        Type::getVoidTy(global_context),
        {Type::getInt64Ty(global_context), Type::getInt32Ty(global_context)},
        false
    );
    FunctionCallee invokeFunc = mainmodule->getOrInsertFunction("InvokeInterpreter", invokeType);
    
    Value* modelHandle = Builder->CreatePtrToInt(modelInstance, Type::getInt64Ty(global_context));
    Builder->CreateCall(invokeFunc, {modelHandle, dummyInt});
    
    return nullptr;
}

DataType ModelNode::getDataType() {
    if (memberName == "output") {
        // Retorna tipo array de float (1 dimensão)
        return buildTypes->getArrayType("float", getLoc(), 1, true);
    }
    // Outros casos podem ser tratados conforme necessário
    return BuildTypes::undefinedType;
}

const std::string ModelNode::getName() const {
    return modelName;
}

