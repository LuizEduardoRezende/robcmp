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

	AiModel(location_t loc): Constructor(loc) {}

public:
	// ai_model is a reserved name: declaring a function, method, type or
	// variable with it is an error, reported here. Returns false if so.
	static bool checkNotReserved(NamedNode *declared);

	// True when fc is a call to the builtin ai_model.
	static bool isBuiltinCall(FunctionCall& fc);

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
