#include "tinyml/AiModel.h"
#include "FunctionCall.h"
#include "StringConst.h"
#include "Scalar.h"
#include "Scanner.h"
#include "tinyml/IntegerConstant.h"
#include "tinyml/WrapperCall.h"
#include "BackLLVM.h"

static const char *builtinName = "ai_model";

// Instances of interpreter in the TFLM wrapper: TFLM_MAX_MODELS, in
// wrappers/tflm/tflm_wrapper.cpp. The two must change together;
// check_ai_model.py fails when they differ.
static const unsigned wrapperInstances = 4;

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

// The calls that declare a model, and whether the variable is global.
static map<const FunctionCall*, bool> declarations;

// Every valid model of the program, in the order the analysis met them.
vector<AiModel::Declared> AiModel::models;

void AiModel::markDeclaration(FunctionCall& fc, bool global) {
	declarations[&fc] = global;
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
	// Having every kernel is not enough: TFLM also refuses some combinations
	// of tensor types (specification, section 8). All the reasons go in one
	// message, since the compiler reports one error per place.
	string missing;
	unsigned count = 0;
	for (const TfliteKernel& k : node->model.kernels) {
		if (k.wrapperFunction.empty()) {
			missing += (missing.empty() ? "" : ", ") + string(k.custom ? "custom " : "builtin ") + k.op;
			count++;
		}
	}
	vector<string> reasons;
	if (count > 0)
		reasons.push_back(string_format("uses %s the TFLM wrapper does not support: %s",
			count == 1 ? "a kernel" : "kernels", missing.c_str()));
	const vector<string>& incompatible = node->model.incompatibilities;
	if (!incompatible.empty()) {
		string what;
		for (const string& why : incompatible)
			what += (what.empty() ? "" : "; ") + why;
		reasons.push_back("can not run on TFLite Micro: " + what +
			(any_of(incompatible.begin(), incompatible.end(), [](const string& w) {
				return w.find(" weights") != string::npos; }) ?
				". Convert the model with full integer quantization" : ""));
	}
	bool supported = reasons.empty();
	if (!supported) {
		string message = "model '" + node->model.path + "' ";
		for (size_t i = 0; i < reasons.size(); i++)
			message += (i > 0 ? ", and " : "") + reasons[i];
		yyerrorcpp(message + ".", path);
	}
	node->valid = supported;
	if (supported)
		models.push_back({node, declarations[&fc]});
	return node;
}

/*
 * A model is created once and never destroyed (specification, sections 4c and
 * 8b), so each one created takes an instance of the wrapper until the program
 * ends. Global models are all created before main: more of them than
 * instances is an error. A local one is created only if its function runs,
 * which the compiler does not know: more models in all than instances is a
 * warning. Checked once, at the code generation, when every declaration has
 * been seen -- a global can be declared below the functions that use it.
 */
void AiModel::checkInstances() {
	static bool checked = false;
	if (checked)
		return;
	checked = true;

	auto before = [](const Declared& a, const Declared& b) {
		AiModel *x = a.model, *y = b.model;
		if (x->getFile() != y->getFile())
			return x->getFile() < y->getFile();
		if (x->getLineNo() != y->getLineNo())
			return x->getLineNo() < y->getLineNo();
		return x->getColNo() < y->getColNo();
	};
	vector<Declared> sorted(models.begin(), models.end());
	std::sort(sorted.begin(), sorted.end(), before);

	unsigned globals = 0;
	for (const Declared& d : sorted) {
		if (d.global && ++globals == wrapperInstances + 1)
			yyerrorcpp(string_format("This is global model %u of the program, but the TFLM "
				"wrapper has %u instances (TFLM_MAX_MODELS): every global model is created "
				"before main runs, and this one can not be.", globals, wrapperInstances),
				d.model);
	}
	if (globals > wrapperInstances || sorted.size() <= wrapperInstances)
		return;
	yywarncpp(string_format("The program declares %u models, and the TFLM wrapper has %u "
		"instances (TFLM_MAX_MODELS). A model is never destroyed: if more than %u of them "
		"are created, the next ones fail at run time, and their invoke() returns -1.",
		(unsigned)sorted.size(), wrapperInstances, wrapperInstances),
		sorted[wrapperInstances].model);
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
	checkInstances();
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
