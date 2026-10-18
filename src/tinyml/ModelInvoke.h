#pragma once

/*
 * Running the inference, in place of the FunctionCall the parser builds for
 * m.invoke(). Returns int8: 0 on success, -1 on failure (specification,
 * section 1).
 */

#include "Node.h"
#include "tinyml/ModelMember.h"

class ModelInvoke: public Node {
private:
	TensorRef model;	// only model and variable are set

public:
	ModelInvoke(const TensorRef& model, location_t loc): Node(loc), model(model) {
		dt = tint8;
	}

	AiModel *getModel() const {
		return model.model;
	}

	Variable *getVariable() const {
		return model.variable;
	}

	virtual Value *generate(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) override;
};
