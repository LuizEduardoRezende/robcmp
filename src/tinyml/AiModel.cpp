#include "tinyml/AiModel.h"
#include "FunctionCall.h"
#include "StringConst.h"
#include "Scalar.h"
#include "Scanner.h"
#include "tinyml/IntegerConstant.h"
#include "tinyml/WrapperCall.h"
#include "BackLLVM.h"

static const char *builtinName = "ai_model";

bool AiModel::checkNotReserved(NamedNode *declared) {
	if (declared->getName() != builtinName)
		return true;
	yyerrorcpp(string_format("%s is a reserved name: it declares a TinyML model.",
		builtinName), declared);
	return false;
}

AiModel *AiModel::heldBy(Node *symbol) {
	Variable *variable = dynamic_cast<Variable*>(symbol);
	return variable ? dynamic_cast<AiModel*>(variable->getExpr()) : NULL;
}

bool AiModel::isBuiltinCall(FunctionCall& fc) {
	return fc.getName() == builtinName;
}

bool AiModel::isPendingDeclaration(Node *symbol) {
	Variable *variable = dynamic_cast<Variable*>(symbol);
	FunctionCall *call = variable ? dynamic_cast<FunctionCall*>(variable->getExpr()) : NULL;
	return call && isBuiltinCall(*call);
}

static set<const FunctionCall*> declarations;

void AiModel::markDeclaration(FunctionCall& fc) {
	declarations.insert(&fc);
}

/*
 * Where a model path is looked for: first the dir of the .rob that has the
 * call, then each -I dir in the order given, as the C #include "...". The
 * order of includeDirs, used by `use`, is a different one.
 */
static vector<filesystem::path> modelCandidates(const string& path, const filesystem::path& callerFile) {
	filesystem::path p(path);
	if (p.is_absolute())
		return {p};

	vector<filesystem::path> candidates;
	filesystem::path callerDir = callerFile.parent_path();
	candidates.push_back(callerDir.empty() ? p : callerDir / p);
	for (auto& dir : cmdlineIncludeDirs)
		candidates.push_back(dir / p);
	return candidates;
}

AiModel *AiModel::fromCall(FunctionCall& fc) {
	AiModel *node = new AiModel(fc.getLoc());
	node->setFileOf(&fc);
	node->setScope(fc.getScope());

	if (!declarations.count(&fc)) {
		yyerrorcpp(string_format("%s(...) declares a model: it is used only to initialize "
			"a new variable, as m = %s(...).", builtinName, builtinName), node);
		return node;
	}

	auto& args = fc.getParameters();
	if (args.size() == 1) {
		yyerrorcpp(string_format("The arena size of %s is missing: "
			"%s(\"model.tflite\", <arena size in bytes>).", builtinName, builtinName), node);
		return node;
	}
	if (args.size() != 2) {
		yyerrorcpp(string_format("%s takes 2 arguments, the model path and the arena size, "
			"but was called with %d.", builtinName, (int)args.size()), node);
		return node;
	}

	StringConst *path = dynamic_cast<StringConst*>(args[0]);
	if (!path) {
		yyerrorcpp(string_format("The model path of %s must be a string literal: "
			"the model is read at compile time.", builtinName), args[0]);
		return node;
	}

	int64_t arena;
	if (!integerConstant(args[1], arena)) {
		yyerrorcpp(string_format("The arena size of %s must be an integer constant.",
			builtinName), args[1]);
		return node;
	}
	if (arena <= 0) {
		yyerrorcpp(string_format("The arena size of %s must be positive, not %lld.",
			builtinName, (long long)arena), args[1]);
		return node;
	}
	node->arenaSize = arena;

	vector<filesystem::path> candidates = modelCandidates(path->getString(), fc.getFile());
	filesystem::path found;
	for (auto& c : candidates) {
		std::error_code ec;
		if (filesystem::is_regular_file(c, ec)) {
			found = c;
			break;
		}
	}
	if (found.empty()) {
		string searched;
		for (auto& c : candidates)
			searched += (searched.empty() ? "" : ", ") + c.string();
		yyerrorcpp(string_format("model '%s': file not found (searched %s)",
			path->getString().c_str(), searched.c_str()), path);
		return node;
	}

	string error;
	if (!TfliteModel::load(found.string(), node->model, error)) {
		yyerrorcpp(error, path);
		return node;
	}

	// The kernels are registered by the code generation, one call for each:
	// a kernel the wrapper can not register would only fail at run time, when
	// the interpreter meets the operator (specification, section 8).
	bool supported = true;
	for (const TfliteKernel& k : node->model.kernels) {
		if (k.wrapperFunction.empty()) {
			yyerrorcpp(string_format("model '%s' uses the %s kernel %s, which the TFLM "
				"wrapper does not support.", node->model.path.c_str(),
				k.custom ? "custom" : "builtin", k.op.c_str()), path);
			supported = false;
		}
	}
	node->valid = supported;
	return node;
}

/*
 * Code generation. For m = ai_model("model.tflite", 3000):
 *
 *   @m.tflite  the bytes of the model, constant (flash, in a MCU)
 *   @m.arena   the arena, zeroed (RAM)
 *   @m.handle  the interpreter, once created
 *
 * both buffers aligned to 16, as the TFLM wants them (RESULTADOS.md, 5.2.8 and
 * 5.4.10), and a function that creates the interpreter -- the three phases of
 * the wrapper, with one tflm_add_* for each kernel the .tflite uses:
 *
 *   uint64 m.ai_model() {
 *       if m.handle != 0 { return m.handle; }
 *       r = tflm_create_resolver();
 *       tflm_add_fully_connected(r);
 *       h = tflm_create_model(r, m.tflite, m.arena);
 *       if h == 0 { tflm_free_model(r); return 0; }
 *       m.handle = h;
 *       return h;
 *   }
 *
 * The declaration calls it. A model is created once (specification, section
 * 4c): a declaration inside a function that runs again gets the same
 * interpreter, instead of a second one over the same arena. A failed creation
 * releases its slot in the wrapper, and the variable holds 0, the handle that
 * the wrapper refuses: invoke() then returns -1.
 */

static GlobalVariable *alignedGlobal(Type *ty, bool constant, Constant *init, const string& name) {
	GlobalVariable *gv = new GlobalVariable(*mainmodule, ty, constant,
		GlobalValue::InternalLinkage, init, name);
	gv->setAlignment(Align(16));
	return gv;
}

Function *AiModel::createFunction(const string& name) {
	IRBuilderBase::InsertPointGuard guard(*Builder);
	Builder->SetCurrentDebugLocation(DebugLoc());

	Type *i32 = Type::getInt32Ty(global_context);
	Type *i64 = Type::getInt64Ty(global_context);
	Constant *zero = ConstantInt::get(i64, 0);

	Constant *bytes = ConstantDataArray::get(global_context, ArrayRef<uint8_t>(model.bytes));
	GlobalVariable *data = alignedGlobal(bytes->getType(), true, bytes, name + ".tflite");
	ArrayType *arenaTy = ArrayType::get(Type::getInt8Ty(global_context), arenaSize);
	GlobalVariable *arena = alignedGlobal(arenaTy, false, ConstantAggregateZero::get(arenaTy),
		name + ".arena");
	GlobalVariable *handle = new GlobalVariable(*mainmodule, i64, false,
		GlobalValue::InternalLinkage, zero, name + ".handle");

	Function *f = Function::Create(FunctionType::get(i64, false), Function::InternalLinkage,
		codeAddrSpace, name + ".ai_model", mainmodule);
	BasicBlock *entry = BasicBlock::Create(global_context, "entry", f);
	BasicBlock *create = BasicBlock::Create(global_context, "create", f);
	BasicBlock *failed = BasicBlock::Create(global_context, "failed", f);
	BasicBlock *created = BasicBlock::Create(global_context, "created", f);
	BasicBlock *existing = BasicBlock::Create(global_context, "existing", f);

	Builder->SetInsertPoint(entry);
	Value *current = Builder->CreateLoad(i64, handle, "current");
	Builder->CreateCondBr(Builder->CreateICmpEQ(current, zero), create, existing);

	Builder->SetInsertPoint(create);
	Value *resolver = WrapperCall::call("tflm_create_resolver", {});
	for (const TfliteKernel& k : model.kernels)
		WrapperCall::call(k.wrapperFunction, {resolver});
	Value *h = WrapperCall::call("tflm_create_model", {resolver, data, arena,
		ConstantInt::get(i32, model.bytes.size()), ConstantInt::get(i32, arenaSize)});
	Builder->CreateCondBr(Builder->CreateICmpEQ(h, zero), failed, created);

	Builder->SetInsertPoint(failed);
	WrapperCall::call("tflm_free_model", {resolver});
	Builder->CreateRet(zero);

	Builder->SetInsertPoint(created);
	Builder->CreateStore(h, handle);
	Builder->CreateRet(h);

	Builder->SetInsertPoint(existing);
	Builder->CreateRet(current);
	return f;
}

Value *AiModel::generate(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) {
	if (!leftValue)
		return NULL;
	Function *create = createFunction(leftValue->getName());

	if (allocblock == global_alloc) {
		// A global model is created before main runs, by __globals_init, as
		// every global (Program::generate). Its variable starts at 0, and the
		// Scalar that declares it stores the handle there.
		Type *i64 = Type::getInt64Ty(global_context);
		leftValue->setAlloca(new GlobalVariable(*mainmodule, i64, false,
			GlobalValue::InternalLinkage, ConstantInt::get(i64, 0), leftValue->getName()));
		Builder->SetInsertPoint(global_alloc);
		return Builder->CreateCall(create);
	}

	Builder->SetInsertPoint(block);
	RobDbgInfo.emitLocation(this);
	return Builder->CreateCall(create);
}
