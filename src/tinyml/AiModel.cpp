#include "tinyml/AiModel.h"
#include "FunctionCall.h"
#include "StringConst.h"
#include "Scalar.h"
#include "Load.h"
#include "NamedConst.h"
#include "Int8.h"
#include "Int16.h"
#include "Int32.h"
#include "Int64.h"
#include "Scanner.h"
#include "BinaryOp.h"
#include "semantic/PropagateTypes.h"

static const char *builtinName = "ai_model";

bool AiModel::checkNotReserved(NamedNode *declared) {
	if (declared->getName() != builtinName)
		return true;
	yyerrorcpp(string_format("%s is a reserved name: it declares a TinyML model.",
		builtinName), declared);
	return false;
}

bool AiModel::isBuiltinCall(FunctionCall& fc) {
	return fc.getName() == builtinName;
}

/*
 * Value of an integer constant: a literal, a const scalar or named constant
 * whose value is one, or +, - and * over them -- enough for 90 * 1024, and for
 * -5, which the parser builds as 5 * -1. Returns false for anything else.
 */
static bool integerConstant(Node *n, int64_t& value) {
	if (auto *i = dynamic_cast<Int8*>(n))   { value = i->getNumber(); return true; }
	if (auto *i = dynamic_cast<UInt8*>(n))  { value = i->getNumber(); return true; }
	if (auto *i = dynamic_cast<Int16*>(n))  { value = i->getNumber(); return true; }
	if (auto *i = dynamic_cast<UInt16*>(n)) { value = i->getNumber(); return true; }
	if (auto *i = dynamic_cast<Int32*>(n))  { value = i->getNumber(); return true; }
	if (auto *i = dynamic_cast<UInt32*>(n)) { value = i->getNumber(); return true; }
	if (auto *i = dynamic_cast<Int64*>(n))  { value = i->getNumber(); return true; }
	if (auto *i = dynamic_cast<UInt64*>(n)) {
		if (i->getNumber() > INT64_MAX)
			return false;
		value = i->getNumber();
		return true;
	}

	if (auto *op = dynamic_cast<BinaryOp*>(n)) {
		int64_t l, r;
		if (!integerConstant(op->lhsn(), l) || !integerConstant(op->rhsn(), r))
			return false;
		switch (op->getOperator()) {
			case '+': return !__builtin_add_overflow(l, r, &value);
			case '-': return !__builtin_sub_overflow(l, r, &value);
			case '*': return !__builtin_mul_overflow(l, r, &value);
			default: return false;
		}
	}
	// a literal coerced to the type of the other operand
	if (auto *coercion = dynamic_cast<CoercionBase*>(n))
		return integerConstant(coercion->children()[0], value);

	if (auto *load = dynamic_cast<Load*>(n))
		return load->isConstExpr() && integerConstant(load->getIdentSymbol(false), value);
	if (auto *scalar = dynamic_cast<Scalar*>(n))
		return scalar->isConstExpr() && integerConstant(scalar->getExpr(), value);
	if (auto *named = dynamic_cast<NamedConst*>(n))
		return named->getValue() && integerConstant(named->getValue(), value);
	return false;
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

	auto& args = fc.getParameters();
	if (args.size() == 1) {
		yyerrorcpp(string_format("The arena size of %s is missing: "
			"%s(\"model.tflite\", <arena size in bytes>).", builtinName, builtinName), node);
		return node;
	}
	if (args.size() != 2) {
		yyerrorcpp(string_format("%s takes 2 arguments, the model path and the arena size, "
			"but was called with %d.", builtinName, args.size()), node);
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

	node->valid = true;
	return node;
}

Value *AiModel::generate(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) {
	yyerrorcpp(string_format("model '%s' was read, but the code generation of %s "
		"is not implemented yet.", model.path.c_str(), builtinName), this);
	return NULL;
}
