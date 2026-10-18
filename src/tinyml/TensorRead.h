#pragma once

/*
 * Reading a tensor of a model, in place of the Load or LoadArray the parser
 * builds:
 *
 *     v = m.output0;          the whole tensor, default view: float[N]
 *     x = m.output0[k];       one element: float
 *     c = m.output0.raw;      the whole tensor, raw view: int8[N] or int16[N]
 *     c = m.output0.raw[k];   one element: int8 or int16
 *
 * Inputs can be read too.
 *
 * The whole tensor is copied into a buffer: a new one, reserved once in the
 * function, for a new variable or a temporary; the vector of the variable, when
 * it already exists (specification, section 4b).
 */

#include "Node.h"
#include "NamedConst.h"
#include "tinyml/ModelMember.h"

class TensorRead: public Node {
private:
	TensorRef tensor;
	Node *element;	// NULL: the whole tensor
	Variable *leftValue = NULL;	// the variable that receives the tensor

public:
	TensorRead(const TensorRef& tensor, Node *element, location_t loc):
		Node(loc), tensor(tensor), element(element) {
		if (element) {
			addChild(element);
			dt = tensor.elementType();
		} else {
			dt = tensor.vectorType(this);
			// the variable that receives it has a .size, as any vector
			NamedConst *size = new NamedConst("size",
				getNodeForUIntConst(tensor.info().elements, loc));
			addChild(size);
			addSymbol(size);
		}
	}

	const TensorRef& getTensor() const {
		return tensor;
	}

	Node *getElement() {
		return element;
	}

	virtual void setLeftValue(Variable *symbol) override {
		leftValue = symbol;
	}

	virtual Value *generate(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) override;
};
