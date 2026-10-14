#include "tinyml/ModelUse.h"
#include "tinyml/ModelMember.h"
#include "tinyml/AiModel.h"
#include "tinyml/TensorRead.h"
#include "tinyml/TensorWrite.h"
#include "tinyml/ModelInvoke.h"
#include "semantic/PropagateTypes.h"
#include "Scalar.h"
#include "UserType.h"
#include "Scanner.h"

// Replaces a use of a model that has an error.
class InvalidModelUse: public Node {
public:
	InvalidModelUse(location_t loc): Node(loc) {
		setSemanticError();
	}
};

static Node *placed(Node *replacement, Node& original) {
	replacement->setScope(original.getScope());
	replacement->setFileOf(&original);
	return replacement;
}

static Node *invalid(Node& original) {
	return placed(new InvalidModelUse(original.getLoc()), original);
}

static Node *fail(Node& original, const string& msg) {
	yyerrorcpp(msg, &original);
	return invalid(original);
}

static Node *propagated(Node *n, PropagateTypes& pt) {
	Node *replacement = n->accept(pt);
	return replacement ? replacement : n;
}

static const char *typeName(DataType dt) {
	return dt == BuildTypes::undefinedType ? "an undefined type" : buildTypes->name(dt);
}

// Errors common to every use that is not a read of a tensor or property.
static Node *notThere(const ModelMember& mm, Node& n, const char *as) {
	switch (mm.kind) {
		case ModelMember::Error:
			mm.report(&n);
			return invalid(n);
		case ModelMember::Model:
			return fail(n, string_format("%s is a model, not %s: use its members -- tensors, "
				"properties and invoke().", mm.name.c_str(), as));
		case ModelMember::Invoke:
			return fail(n, string_format("%s is the method of the model that runs the "
				"inference: call it as %s().", mm.name.c_str(), mm.name.c_str()));
		default:
			return fail(n, string_format("%s is not %s.", mm.name.c_str(), as));
	}
}

static Node *writable(const ModelMember& mm, Node& n) {
	if (mm.kind == ModelMember::Property)
		return fail(n, string_format("%s is read-only.", mm.name.c_str()));
	if (mm.kind == ModelMember::Model)
		return fail(n, string_format("%s already holds a model: a model is declared once, "
			"and can not be assigned.", mm.name.c_str()));
	if (mm.kind != ModelMember::Tensor)
		return notThere(mm, n, "a variable");
	if (mm.tensor.output)
		return fail(n, string_format("%s is an output tensor: it is read-only.", mm.name.c_str()));
	return NULL;
}

// The whole-tensor value must be a vector of the element type of the view.
static Node *checkVector(const ModelMember& mm, Node *value, Node& n) {
	DataType vdt = value->getDataType();
	DataType edt = mm.tensor.elementType();
	if (buildTypes->isArray(vdt) && buildTypes->getArrayElementType(vdt) == edt)
		return NULL;
	if (!mm.tensor.raw)
		return fail(n, string_format("%s takes a vector of real values, float[], not %s. "
			"For codes in the representation of the tensor, use %s.raw.", mm.name.c_str(),
			typeName(vdt), mm.name.c_str()));
	return fail(n, string_format("The raw view %s takes %s[], the type of the tensor, not %s.",
		mm.name.c_str(), buildTypes->name(edt), typeName(vdt)));
}

Node *ModelUse::load(Load& n, PropagateTypes& pt) {
	ModelMember mm = ModelMember::resolve(n.getName(), n.getScope());
	switch (mm.kind) {
		case ModelMember::None:
			return NULL;
		case ModelMember::Property:
			return placed(mm.constant(n.getLoc()), n);
		case ModelMember::Tensor:
			return placed(new TensorRead(mm.tensor, NULL, n.getLoc()), n);
		default:
			return notThere(mm, n, "a value");
	}
}

Node *ModelUse::loadArray(LoadArray& n, PropagateTypes& pt) {
	ModelMember mm = ModelMember::resolve(n.getName(), n.getScope());
	if (mm.kind == ModelMember::None)
		return NULL;
	if (mm.kind != ModelMember::Tensor)
		return notThere(mm, n, "a vector");

	Node *element = propagated(n.getPosition(), pt);
	return placed(new TensorRead(mm.tensor, element, n.getLoc()), n);
}

Node *ModelUse::updateArray(UpdateArray& n, PropagateTypes& pt) {
	ModelMember mm = ModelMember::resolve(n.getName(), n.getScope());
	if (mm.kind == ModelMember::None)
		return NULL;
	if (mm.kind != ModelMember::Tensor)
		return notThere(mm, n, "a vector");
	if (Node *error = writable(mm, n))
		return error;

	Node *element = propagated(n.getPosition(), pt);
	Node *value = PropagateTypes::coerceTo(propagated(n.getExpr(), pt), mm.tensor.elementType());
	if (!value)
		return invalid(n);
	return placed(new TensorWrite(mm.tensor, element, value, n.getLoc()), n);
}

Node *ModelUse::scalar(Scalar& n, PropagateTypes& pt) {
	// A model is declared in a new variable: m = ai_model(...) where m does
	// not exist yet.
	FunctionCall *call = dynamic_cast<FunctionCall*>(n.getExpr());
	if (call && AiModel::isBuiltinCall(*call)) {
		AiModel::markDeclaration(*call);

		// Two instances of the type would share the model and its arena; how
		// to store them is decided with the code generation (task 3.6).
		// The declaration goes on, so that its uses report nothing more.
		if (dynamic_cast<UserType*>(n.getScope())) {
			yyerrorcpp(string_format("A model can not be a field of a type: declare "
				"%s in a function or as a global.", n.getName().c_str()), &n);
			return NULL;
		}
		Node *first = n.getScope()->findSymbol(n.getName());
		if (first && first != &n && !n.getIdent().isComplex())
			return fail(n, string_format("A model is declared in a new variable: "
				"%s already exists.", n.getName().c_str()));
		return NULL;
	}

	ModelMember mm = ModelMember::resolve(n.getName(), n.getScope());
	if (mm.kind == ModelMember::None)
		return NULL;
	if (Node *error = writable(mm, n))
		return error;

	Node *value = propagated(n.getExpr(), pt);
	if (Node *error = checkVector(mm, value, n))
		return error;
	return placed(new TensorWrite(mm.tensor, NULL, value, n.getLoc()), n);
}

Node *ModelUse::array(Array& n, PropagateTypes& pt) {
	ModelMember mm = ModelMember::resolve(n.getName(), n.getScope());
	if (mm.kind == ModelMember::None)
		return NULL;
	if (Node *error = writable(mm, n))
		return error;

	// The default view takes real values, whatever the type of the literals:
	// {1} is the value 1.0. The raw view takes the exact type of the tensor
	// (specification, section 4b).
	DataType edt = mm.tensor.elementType();
	for (ArrayElement *ae : n.getElements()) {
		ae->value = propagated(ae->value, pt);
		if (!mm.tensor.raw) {
			ae->value = PropagateTypes::coerceTo(ae->value, edt);
			if (!ae->value)
				return invalid(n);
		}
	}
	if (Node *error = checkVector(mm, &n, n))
		return error;
	return placed(new TensorWrite(mm.tensor, NULL, &n, n.getLoc()), n);
}

Node *ModelUse::call(FunctionCall& n, PropagateTypes& pt) {
	ModelMember mm = ModelMember::resolve(n.getName(), n.getScope());
	if (mm.kind == ModelMember::None)
		return NULL;
	if (mm.kind != ModelMember::Invoke)
		return notThere(mm, n, "a function: the only method of a model is invoke()");
	if (n.getParameters().size() > 0)
		return fail(n, string_format("%s() takes no arguments.", mm.name.c_str()));
	return placed(new ModelInvoke(mm.tensor, n.getLoc()), n);
}
