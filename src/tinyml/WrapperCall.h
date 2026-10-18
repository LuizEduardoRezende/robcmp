#pragma once

/*
 * Calls to the TFLM wrapper (wrappers/tflm/tflm_wrapper.h) emitted by the code
 * generation of the native syntax.
 *
 * Each function is declared exactly as the library mode declares it in
 * lib/ai/tflm.rob, so that both modes call the wrapper with the same IR: a
 * program that uses the two shares one declaration, and the signatures that
 * `make -C test/tflm abi` checks are the ones used here.
 */

#include "HeaderGlobals.h"

class Node;
class FunctionImpl;
class Variable;
struct TensorRef;

class WrapperCall {
public:
	// Calls a function of the wrapper at the insert point of the Builder. The
	// size of each vector argument goes after all the others, as the robcmp
	// passes them: tflm_create_model(r, model, arena, model size, arena size).
	static CallInst *call(const string& name, ArrayRef<Value*> args);

	// The handle of the model held by variable.
	static Value *modelHandle(Variable *variable, FunctionImpl *func);

	// The handle of a tensor: tflm_input(m, i) or tflm_output(m, i).
	static Value *tensorHandle(const TensorRef& tensor, FunctionImpl *func);

	// The number of elements of a vector, as a call to a function with a
	// vector parameter passes it (FunctionCall::generate): its .size.
	static Value *vectorSize(Node *vector, FunctionImpl *func, BasicBlock *block,
		BasicBlock *allocblock);

	// An integer value as the uint64 of an index.
	static Value *index(Node *element, FunctionImpl *func, BasicBlock *block,
		BasicBlock *allocblock);
};
