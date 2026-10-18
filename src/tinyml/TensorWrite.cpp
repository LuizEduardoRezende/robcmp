#include "tinyml/TensorWrite.h"
#include "tinyml/WrapperCall.h"
#include "BackLLVM.h"

static const char *writeFunction(const TensorRef& tensor) {
	if (!tensor.raw)
		return "tflm_set_values";
	return tensor.info().type == TfliteType::Int8 ? "tflm_set_raw_int8" : "tflm_set_raw_int16";
}

Value *TensorWrite::generate(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) {
	if (allocblock == global_alloc) {
		yyerrorcpp(string_format("A tensor is written inside a function: %s can not "
			"be written in the global scope.", tensor.name().c_str()), this);
		return NULL;
	}

	Value *index = element ? WrapperCall::index(element, func, block, allocblock) : NULL;
	Value *v = value->generate(func, block, allocblock);
	if (!v)
		return NULL;
	Value *size = element ? NULL : WrapperCall::vectorSize(value, func, block, allocblock);

	Builder->SetInsertPoint(block);
	RobDbgInfo.emitLocation(this);
	Value *t = WrapperCall::tensorHandle(tensor, func);
	if (!element)
		return WrapperCall::call(writeFunction(tensor), {t, v, size});
	if (!tensor.raw)
		return WrapperCall::call("tflm_set_value", {t, index, v});
	// the code goes as int32 (wrappers/tflm/tflm_wrapper.h)
	Value *code = Builder->CreateSExt(v, Type::getInt32Ty(global_context));
	return WrapperCall::call("tflm_set_raw_value", {t, index, code});
}
