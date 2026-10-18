#include "tinyml/WrapperCall.h"
#include "tinyml/ModelMember.h"
#include "tinyml/AiModel.h"
#include "semantic/PropagateTypes.h"
#include "FunctionDecl.h"
#include "FunctionImpl.h"
#include "BackLLVM.h"
#include "Program.h"
#include "Load.h"

// The types of lib/ai/tflm.rob, as the robcmp declares them.
enum class Arg {
	Void,
	Handle,		// uint64: handle or index, zeroext
	Int8,		// int8
	Int32,		// int32
	Float,		// float
	Vector		// T[]: the address of the first element, then the size, uint32 zeroext
};

struct Signature {
	Arg ret;
	vector<Arg> params;
};

static const Signature *signatureOf(const string& name) {
	static const map<string, Signature> signatures = {
		{"tflm_create_resolver", {Arg::Handle, {}}},
		{"tflm_create_model",    {Arg::Handle, {Arg::Handle, Arg::Vector, Arg::Vector}}},
		{"tflm_free_model",      {Arg::Void,   {Arg::Handle}}},
		{"tflm_invoke",          {Arg::Int8,   {Arg::Handle}}},
		{"tflm_input",           {Arg::Handle, {Arg::Handle, Arg::Handle}}},
		{"tflm_output",          {Arg::Handle, {Arg::Handle, Arg::Handle}}},
		{"tflm_set_value",       {Arg::Void,   {Arg::Handle, Arg::Handle, Arg::Float}}},
		{"tflm_get_value",       {Arg::Float,  {Arg::Handle, Arg::Handle}}},
		{"tflm_set_values",      {Arg::Int8,   {Arg::Handle, Arg::Vector}}},
		{"tflm_get_values",      {Arg::Int8,   {Arg::Handle, Arg::Vector}}},
		{"tflm_set_raw_int8",    {Arg::Int8,   {Arg::Handle, Arg::Vector}}},
		{"tflm_get_raw_int8",    {Arg::Int8,   {Arg::Handle, Arg::Vector}}},
		{"tflm_set_raw_int16",   {Arg::Int8,   {Arg::Handle, Arg::Vector}}},
		{"tflm_get_raw_int16",   {Arg::Int8,   {Arg::Handle, Arg::Vector}}},
		{"tflm_set_raw_value",   {Arg::Void,   {Arg::Handle, Arg::Handle, Arg::Int32}}},
		{"tflm_get_raw_value",   {Arg::Int32,  {Arg::Handle, Arg::Handle}}},
	};
	// every kernel: int8 tflm_add_<kernel>(uint64 resolver_handle)
	static const Signature addKernel = {Arg::Int8, {Arg::Handle}};

	auto s = signatures.find(name);
	if (s != signatures.end())
		return &s->second;
	if (name.rfind("tflm_add_", 0) == 0)
		return &addKernel;
	return nullptr;
}

static Type *llvmType(Arg a) {
	switch (a) {
		case Arg::Void:   return Type::getVoidTy(global_context);
		case Arg::Handle: return Type::getInt64Ty(global_context);
		case Arg::Int8:   return Type::getInt8Ty(global_context);
		case Arg::Int32:  return Type::getInt32Ty(global_context);
		case Arg::Float:  return Type::getFloatTy(global_context);
		case Arg::Vector: return PointerType::get(global_context, 0);
	}
	return nullptr;
}

static Function *declare(const string& name) {
	// Declared by `use ai.tflm`: the same function, so that the two modes do
	// not create two declarations of one symbol.
	if (FunctionDecl *decl = dynamic_cast<FunctionDecl*>(program->findSymbol(name)))
		return dyn_cast<Function>(decl->getLLVMValue(NULL, NULL));
	if (Function *f = mainmodule->getFunction(name))
		return f;

	const Signature *s = signatureOf(name);
	assert(s && "not a function of the wrapper");
	// The size of each vector goes after all the parameters, in their order.
	vector<Type*> params;
	vector<bool> zeroext;
	for (Arg a : s->params) {
		params.push_back(llvmType(a));
		zeroext.push_back(a == Arg::Handle);
	}
	for (Arg a : s->params) {
		if (a == Arg::Vector) {
			params.push_back(Type::getInt32Ty(global_context));
			zeroext.push_back(true);
		}
	}
	FunctionType *ft = FunctionType::get(llvmType(s->ret), params, false);
	Function *f = Function::Create(ft, Function::ExternalLinkage, codeAddrSpace, name, mainmodule);
	f->setCallingConv(CallingConv::C);
	if (s->ret == Arg::Handle)
		f->addRetAttr(Attribute::ZExt);
	for (unsigned i = 0; i < zeroext.size(); i++)
		if (zeroext[i])
			f->getArg(i)->addAttr(Attribute::ZExt);
	return f;
}

CallInst *WrapperCall::call(const string& name, ArrayRef<Value*> args) {
	Function *f = declare(name);
	CallInst *call = Builder->CreateCall(f, args);
	call->setAttributes(f->getAttributes());
	return call;
}

Value *WrapperCall::modelHandle(Variable *variable, FunctionImpl *func) {
	Value *alloc = variable->getLLVMValue(func);
	return Builder->CreateLoad(Type::getInt64Ty(global_context), alloc, variable->getName());
}

Value *WrapperCall::tensorHandle(const TensorRef& tensor, FunctionImpl *func) {
	Value *model = modelHandle(tensor.variable, func);
	Value *index = ConstantInt::get(Type::getInt64Ty(global_context), tensor.index);
	return call(tensor.output ? "tflm_output" : "tflm_input", {model, index});
}

Value *WrapperCall::vectorSize(Node *vector, FunctionImpl *func, BasicBlock *block,
	BasicBlock *allocblock) {
	if (Node *size = vector->findMember("size")) {
		Node *coerced = PropagateTypes::coerceTo(size, tint32u);
		if (Value *v = coerced->generate(func, block, allocblock))
			return v;
	}
	Load ld(Identifier(vector->getName() + ".size", vector->getLoc()));
	ld.setScope(func);
	return PropagateTypes::coerceTo(&ld, tint32u)->generate(func, block, allocblock);
}

Value *WrapperCall::index(Node *element, FunctionImpl *func, BasicBlock *block,
	BasicBlock *allocblock) {
	return PropagateTypes::coerceTo(element, tint64u)->generate(func, block, allocblock);
}
