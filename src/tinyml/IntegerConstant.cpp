#include "tinyml/IntegerConstant.h"
#include "Scalar.h"
#include "Load.h"
#include "NamedConst.h"
#include "Int8.h"
#include "Int16.h"
#include "Int32.h"
#include "Int64.h"
#include "BinaryOp.h"
#include "semantic/PropagateTypes.h"

/*
 * Value of an integer constant: a literal, a const scalar or named constant
 * whose value is one, or +, - and * over them -- enough for 90 * 1024, and for
 * -5, which the parser builds as 5 * -1. Returns false for anything else.
 */
bool integerConstant(Node *n, int64_t& value) {
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

bool staticSize(Node *vector, int64_t& size) {
	auto& symbols = vector->getSymbols();
	auto it = symbols.find("size");
	if (it == symbols.end())
		return false;
	NamedConst *named = dynamic_cast<NamedConst*>(it->second);
	return named && named->getValue() && integerConstant(named->getValue(), size);
}
