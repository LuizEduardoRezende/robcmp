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
 * Inputs can be read too. The code generation is task 3.6.
 */

#include "Node.h"
#include "NamedConst.h"
#include "tinyml/ModelMember.h"

class TensorRead: public Node {
private:
	TensorRef tensor;
	Node *element;	// NULL: the whole tensor

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

	virtual Value *generate(FunctionImpl *func, BasicBlock *block, BasicBlock *allocblock) override {
		// task 3.6; never reached until then (see AiModel::fromCall)
		return NULL;
	}
};
