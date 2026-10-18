#include "tinyml/TensorRead.h"
#include "tinyml/WrapperCall.h"
#include "FunctionImpl.h"
#include "BackLLVM.h"

static const char *readFunction(const TensorRef& tensor) {
	if (!tensor.raw)
		return "tflm_get_values";
	return tensor.info().type == TfliteType::Int8 ? "tflm_get_raw_int8" : "tflm_get_raw_int16";
}

Value *TensorRead::generate(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) {
	if (allocblock == global_alloc) {
		yyerrorcpp(string_format("A tensor is read inside a function: %s can not "
			"initialize a global.", tensor.name().c_str()), this);
		return NULL;
	}

	if (element) {
		Value *index = WrapperCall::index(element, func, block, allocblock);
		Builder->SetInsertPoint(block);
		RobDbgInfo.emitLocation(this);
		Value *t = WrapperCall::tensorHandle(tensor, func);
		if (!tensor.raw)
			return WrapperCall::call("tflm_get_value", {t, index});
		Value *code = WrapperCall::call("tflm_get_raw_value", {t, index});
		return Builder->CreateTrunc(code, buildTypes->llvmType(dt));
	}

	Builder->SetInsertPoint(block);
	RobDbgInfo.emitLocation(this);

	// The variable that receives the tensor already exists: the tensor is
	// copied into its vector, not assigned to it (RESULTADOS.md, 5.2.18).
	Value *existing = leftValue ? leftValue->getLLVMValue(func) : NULL;
	Value *buffer, *size, *result;
	if (existing) {
		buffer = existing;
		if (leftValue->isPointerToPointer())
			buffer = Builder->CreateLoad(PointerType::get(global_context, 0), existing, "deref");
		size = WrapperCall::vectorSize(leftValue, func, block, allocblock);
		Builder->SetInsertPoint(block);
		result = existing;
	} else {
		// A new buffer, reserved once in the function and refilled each time
		// the read runs. A new variable refers to it, as to any vector.
		uint64_t elements = tensor.info().elements;
		Type *bufferTy = ArrayType::get(buildTypes->llvmType(tensor.elementType()), elements);
		Builder->SetInsertPoint(allocblock);
		buffer = Builder->CreateAlloca(bufferTy, dataAddrSpace, 0,
			leftValue ? leftValue->getName() : tensor.name());
		Builder->SetInsertPoint(block);
		size = ConstantInt::get(Type::getInt32Ty(global_context), elements);
		if (leftValue)
			leftValue->setPointerToPointer(true);
		result = buffer;
	}

	Value *t = WrapperCall::tensorHandle(tensor, func);
	WrapperCall::call(readFunction(tensor), {t, buffer, size});
	return result;
}
