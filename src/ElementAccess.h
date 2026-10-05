#pragma once

/*
 * Syntactic forms that continue after an indexed element:
 *
 *   id[i] = {...};         UpdateElementArray   array literal into an element
 *   x = id[i].member       LoadElementMember    member read after an index
 *   id[i].member = v;      UpdateElementMember  member write after an index
 *   id[i].member = {...};  (same node, with an array literal)
 *
 * The parser cannot tell what `id` is: types are only known in the semantic
 * analysis. These nodes record the form; the semantic analysis gives them a
 * meaning or rejects them. Today their only meaning is on ai_model tensors
 * (e.g. m.input[0] = {1.0}, m.input[0].scale, m.input[0].raw = v), and every
 * other use is a semantic error -- before these rules, a syntax error.
 */

#include "Node.h"
#include "LoadArray.h"
#include "ArrayElements.h"
#include "semantic/Visitor.h"

class ElementAccess: public BaseArrayOper {
public:
	ElementAccess(const string &i, Node *pos, location_t loc);

	// Text of the accessed element as written, e.g. "m.input[...]".
	virtual string describe() const;
};

class UpdateElementArray: public ElementAccess {
protected:
	ArrayElements *elements;
public:
	UpdateElementArray(const string &i, Node *pos, ArrayElements *elements, location_t loc);

	ArrayElements *getElements() {
		return elements;
	}

	virtual Node* accept(Visitor& v) override {
		return v.visit(*this);
	}
};

class LoadElementMember: public ElementAccess {
protected:
	string member;
public:
	LoadElementMember(const string &i, Node *pos, const string &member, location_t loc);

	const string& getMember() const {
		return member;
	}

	virtual string describe() const override;

	virtual Node* accept(Visitor& v) override {
		return v.visit(*this);
	}
};

class UpdateElementMember: public ElementAccess {
protected:
	string member;
	Node *expr = nullptr;					// id[i].member = expr;
	ArrayElements *elements = nullptr;		// id[i].member = {...};
public:
	UpdateElementMember(const string &i, Node *pos, const string &member, Node *expr, location_t loc);
	UpdateElementMember(const string &i, Node *pos, const string &member, ArrayElements *elements, location_t loc);

	const string& getMember() const {
		return member;
	}

	Node *getExpr() {
		return expr;
	}

	ArrayElements *getElements() {
		return elements;
	}

	virtual string describe() const override;

	virtual Node* accept(Visitor& v) override {
		return v.visit(*this);
	}
};
