#pragma once

/*
 * Writing a tensor of a model, in place of the Scalar, Array or UpdateArray the
 * parser builds:
 *
 *     m.input0 = {1.57};       the whole tensor, default view: float values
 *     m.input0 = v;            v: float[]
 *     m.input0[k] = x;         one element, converted to float
 *     m.input0.raw = codes;    the whole tensor, raw view: int8[] or int16[]
 *     m.input0.raw[k] = c;     one element, converted to the tensor type
 *
 * Only inputs can be written (specification, section 4c). The code
 * generation is task 3.6.
 */

#include "Node.h"
#include "tinyml/ModelMember.h"

class TensorWrite: public Node {
private:
	TensorRef tensor;
	Node *element;	// NULL: the whole tensor
	Node *value;

public:
	TensorWrite(const TensorRef& tensor, Node *element, Node *value, location_t loc):
		Node(loc), tensor(tensor), element(element), value(value) {
		if (element)
			addChild(element);
		addChild(value);
	}

	const TensorRef& getTensor() const {
		return tensor;
	}

	Node *getElement() {
		return element;
	}

	Node *getValue() {
		return value;
	}

	virtual Value *generate(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) override {
		// task 3.6; never reached until then (see AiModel::fromCall)
		return NULL;
	}
};
