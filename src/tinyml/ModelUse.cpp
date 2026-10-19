#include "tinyml/ModelUse.h"
#include "tinyml/ModelMember.h"
#include "tinyml/AiModel.h"
#include "tinyml/TensorRead.h"
#include "tinyml/TensorWrite.h"
#include "tinyml/ModelInvoke.h"
#include "tinyml/IntegerConstant.h"
#include "semantic/PropagateTypes.h"
#include "Scalar.h"
#include "UserType.h"
#include "Scanner.h"
#include "Program.h"

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

static const char *plural(uint64_t n) {
	return n == 1 ? "" : "s";
}

// A vector whose size is known at compile time must have exactly the number
// of elements of the tensor (specification, sections 4b and 8). A vector of
// unknown size, a parameter float[], is checked by the wrapper at run time.
static Node *checkSize(const ModelMember& mm, Node *value, Node& n) {
	int64_t size;
	uint64_t elements = mm.tensor.info().elements;
	if (!staticSize(value, size) || (uint64_t)size == elements)
		return NULL;
	return fail(n, string_format("%s has %llu element%s; the vector has %lld.", mm.name.c_str(),
		(unsigned long long)elements, plural(elements), (long long)size));
}

// A constant element index must be inside the tensor (specification, section
// 4). A variable index is checked by the wrapper at run time.
static Node *checkIndex(const ModelMember& mm, Node *element, Node& n) {
	int64_t index;
	uint64_t elements = mm.tensor.info().elements;
	if (!integerConstant(element, index) || (index >= 0 && (uint64_t)index < elements))
		return NULL;
	return fail(n, string_format("%s has %llu element%s: index %lld is out of the tensor.",
		mm.name.c_str(), (unsigned long long)elements, plural(elements), (long long)index));
}

// saida = m.output0 where saida already exists: the tensor is copied into the
// existing vector (specification, section 4b), which must have the type of the
// view and, when known, the number of elements of the tensor.
static Node *checkReadInto(Scalar& n, PropagateTypes& pt) {
	Load *load = dynamic_cast<Load*>(n.getExpr());
	if (!load)
		return NULL;
	Node *target = n.getIdent().getSymbol(n.getScope(), false);
	if (!target || target == &n)
		return NULL;	// a new variable takes the size of the tensor

	ModelMember read = ModelMember::resolve(load->getName(), load->getScope(), pt);
	if (read.kind != ModelMember::Tensor)
		return NULL;

	DataType vdt = read.tensor.vectorType(load);
	DataType tdt = target->getDataType();
	if (tdt != vdt)
		return fail(n, string_format("%s is %s, and %s, which receives it, is %s.",
			read.name.c_str(), typeName(vdt), n.getName().c_str(), typeName(tdt)));

	int64_t size;
	uint64_t elements = read.tensor.info().elements;
	if (staticSize(target, size) && (uint64_t)size != elements)
		return fail(n, string_format("%s has %llu element%s, and %s, which receives it, has %lld.",
			read.name.c_str(), (unsigned long long)elements, plural(elements),
			n.getName().c_str(), (long long)size));
	return NULL;
}

Node *ModelUse::load(Load& n, PropagateTypes& pt) {
	ModelMember mm = ModelMember::resolve(n.getName(), n.getScope(), pt);
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
	ModelMember mm = ModelMember::resolve(n.getName(), n.getScope(), pt);
	if (mm.kind == ModelMember::None)
		return NULL;
	if (mm.kind != ModelMember::Tensor)
		return notThere(mm, n, "a vector");

	Node *element = propagated(n.getPosition(), pt);
	if (Node *error = checkIndex(mm, element, n))
		return error;
	return placed(new TensorRead(mm.tensor, element, n.getLoc()), n);
}

Node *ModelUse::updateArray(UpdateArray& n, PropagateTypes& pt) {
	ModelMember mm = ModelMember::resolve(n.getName(), n.getScope(), pt);
	if (mm.kind == ModelMember::None)
		return NULL;
	if (mm.kind != ModelMember::Tensor)
		return notThere(mm, n, "a vector");
	if (Node *error = writable(mm, n))
		return error;

	Node *element = propagated(n.getPosition(), pt);
	if (Node *error = checkIndex(mm, element, n))
		return error;
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
		AiModel::markDeclaration(*call, dynamic_cast<Program*>(n.getScope()) != NULL);

		// Two instances of the type would share the model and its arena; how
		// to store them is an open decision (RESULTADOS.md).
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

	// A global declaration visited earlier, when a use further up in the
	// file needed it (ModelMember::resolve)
	if (dynamic_cast<AiModel*>(n.getExpr()))
		return NULL;

	ModelMember mm = ModelMember::resolve(n.getName(), n.getScope(), pt);
	if (mm.kind == ModelMember::None)
		return checkReadInto(n, pt);
	if (Node *error = writable(mm, n))
		return error;

	Node *value = propagated(n.getExpr(), pt);
	if (Node *error = checkVector(mm, value, n))
		return error;
	if (Node *error = checkSize(mm, value, n))
		return error;
	return placed(new TensorWrite(mm.tensor, NULL, value, n.getLoc()), n);
}

Node *ModelUse::array(Array& n, PropagateTypes& pt) {
	ModelMember mm = ModelMember::resolve(n.getName(), n.getScope(), pt);
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
	if (Node *error = checkSize(mm, &n, n))
		return error;
	return placed(new TensorWrite(mm.tensor, NULL, &n, n.getLoc()), n);
}

Node *ModelUse::call(FunctionCall& n, PropagateTypes& pt) {
	ModelMember mm = ModelMember::resolve(n.getName(), n.getScope(), pt);
	if (mm.kind == ModelMember::None)
		return NULL;
	if (mm.kind != ModelMember::Invoke)
		return notThere(mm, n, "a function: the only method of a model is invoke()");
	if (n.getParameters().size() > 0)
		return fail(n, string_format("%s() takes no arguments.", mm.name.c_str()));
	return placed(new ModelInvoke(mm.tensor, n.getLoc()), n);
}
