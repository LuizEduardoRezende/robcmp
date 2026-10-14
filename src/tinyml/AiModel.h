#pragma once

/*
 * Declaration of a TinyML model in the native syntax:
 *
 *     classifier = ai_model("models/hello_world_float.tflite", 3000);
 *
 * The parser reads this as an ordinary function call. PropagateTypes replaces
 * the FunctionCall by this node, which checks the arguments and reads the model
 * at compile time. The members of the model are typed from getModel()
 * (task 3.4), and the code generation embeds its bytes and registers its
 * kernels (task 3.6).
 */

#include "MemCopy.h"
#include "tinyml/TfliteModel.h"

class FunctionCall;

class AiModel: public Constructor {
private:
	TfliteModel model;
	uint64_t arenaSize = 0;
	bool valid = false;

	// The variable holds the handle of the TFLM interpreter, as the
	// tflm_create_model of the library mode returns. It is never seen as a
	// number: the model is not a value (specification, section 4c).
	AiModel(location_t loc): Constructor(loc) {
		dt = tint64u;
	}

public:
	// ai_model is a reserved name: declaring a function, method, type or
	// variable with it is an error, reported here. Returns false if so.
	static bool checkNotReserved(NamedNode *declared);

	// True when fc is a call to the builtin ai_model.
	static bool isBuiltinCall(FunctionCall& fc);

	// fc initializes a variable: m = ai_model(...). Any other use of
	// ai_model(...) is an error, since a model is not a value
	// (specification, section 4c).
	static void markDeclaration(FunctionCall& fc);

	// The model held by a variable declared as m = ai_model(...), or NULL.
	static AiModel *heldBy(Node *symbol);

	// Checks the arguments and reads the model. An error is reported at the
	// call, and the node returned is marked invalid; it still replaces the
	// call, so that it is not reported again as an undefined function.
	static AiModel *fromCall(FunctionCall& fc);

	bool isValid() const {
		return valid;
	}

	const TfliteModel& getModel() const {
		return model;
	}

	uint64_t getArenaSize() const {
		return arenaSize;
	}

	virtual Value *generate(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) override;
};
