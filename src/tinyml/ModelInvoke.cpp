#include "tinyml/ModelInvoke.h"
#include "tinyml/WrapperCall.h"
#include "FunctionImpl.h"

Value *ModelInvoke::generate(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) {
	if (allocblock == global_alloc) {
		yyerrorcpp(string_format("The inference runs inside a function: %s.invoke() can "
			"not initialize a global.", getVariable()->getName().c_str()), this);
		return NULL;
	}
	Builder->SetInsertPoint(block);
	RobDbgInfo.emitLocation(this);
	return WrapperCall::call("tflm_invoke", {WrapperCall::modelHandle(getVariable(), func)});
}
