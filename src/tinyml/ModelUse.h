#pragma once

/*
 * Uses of a model, recognized during PropagateTypes.
 *
 * The parser reads m.input0, m.output0[k], m.input0 = v and m.invoke() as
 * ordinary loads, stores and calls. Each function below receives the node the
 * parser built, and when its name is a member of a model (ModelMember)
 * returns the node that replaces it: a literal for a property, TensorRead,
 * TensorWrite or ModelInvoke for an operation. It returns NULL when the name
 * is not a member of a model, and the compiler goes on as before.
 *
 * On an error, the replacement is an invalid node, so that the name is not
 * reported again as an unknown symbol.
 */

class Node;
class Load;
class LoadArray;
class UpdateArray;
class Scalar;
class Array;
class FunctionCall;
class PropagateTypes;

class ModelUse {
public:
	static Node *load(Load& n, PropagateTypes& pt);
	static Node *loadArray(LoadArray& n, PropagateTypes& pt);
	static Node *updateArray(UpdateArray& n, PropagateTypes& pt);
	static Node *scalar(Scalar& n, PropagateTypes& pt);
	static Node *array(Array& n, PropagateTypes& pt);
	static Node *call(FunctionCall& n, PropagateTypes& pt);
};
