#include "tinyml/ModelMember.h"
#include "tinyml/AiModel.h"
#include "Variable.h"
#include "Float.h"
#include "Scanner.h"
#include "Program.h"
#include "semantic/PropagateTypes.h"

#include <regex>

const TfliteTensor& TensorRef::info() const {
	const TfliteModel& m = model->getModel();
	return output ? m.outputs[index] : m.inputs[index];
}

string TensorRef::name() const {
	return string(output ? "output" : "input") + to_string(index) + (raw ? ".raw" : "");
}

DataType TensorRef::elementType() const {
	if (!raw)
		return tfloat;
	return info().type == TfliteType::Int8 ? tint8 : tint16;
}

DataType TensorRef::vectorType(SourceLocation *loc) const {
	return buildTypes->getArrayType(buildTypes->name(elementType()), *loc, 1, true);
}

static vector<string> split(const string& name) {
	vector<string> parts;
	stringstream ss(name);
	string part;
	while (getline(ss, part, '.'))
		parts.push_back(part);
	return parts;
}

static string join(const vector<string>& parts, size_t from, size_t to) {
	string r;
	for (size_t i = from; i < to; i++)
		r += (i > from ? "." : "") + parts[i];
	return r;
}

// "input0, input1"
static string tensorList(const vector<TfliteTensor>& tensors, const char *role) {
	string r;
	for (size_t i = 0; i < tensors.size(); i++)
		r += (i > 0 ? ", " : "") + string(role) + to_string(i);
	return r;
}

ModelMember ModelMember::resolve(const string& name, Node *scope, PropagateTypes& pt) {
	ModelMember mm;
	mm.name = name;

	// The shortest stem that holds a model: m in m.input0.size, self.m in
	// self.m.input0.
	vector<string> parts = split(name);
	size_t stemEnd = 0;
	for (size_t k = 1; k <= parts.size() && !mm.tensor.model; k++) {
		Identifier stem(join(parts, 0, k), {0, 0, 0, 0});
		Node *symbol = stem.getSymbol(scope, false);

		// The declaration was not visited yet. A global one is declared
		// further down in the file, and is visited now: globals exist before
		// any function runs, whatever their order. A local one is used
		// before it exists, as any local variable would be.
		if (AiModel::isPendingDeclaration(symbol)) {
			if (!dynamic_cast<Program*>(symbol->getScope())) {
				mm.kind = Error;
				mm.error = string_format("%s is used before its declaration: declare the "
					"model, %s = ai_model(...), before using it.", stem.getFullName().c_str(),
					stem.getFullName().c_str());
				return mm;
			}
			symbol->accept(pt);
		}

		if (AiModel *model = AiModel::heldBy(symbol)) {
			mm.tensor.model = model;
			mm.tensor.variable = dynamic_cast<Variable*>(symbol);
			stemEnd = k;
		}
	}
	if (!mm.tensor.model)
		return mm;

	string stem = join(parts, 0, stemEnd);
	vector<string> rest(parts.begin() + stemEnd, parts.end());
	if (!mm.tensor.model->isValid()) {
		// the declaration already has an error
		mm.kind = Error;
		return mm;
	}

	const TfliteModel& model = mm.tensor.model->getModel();
	auto fail = [&](const string& msg) {
		mm.kind = Error;
		mm.error = msg;
		return mm;
	};

	if (rest.empty()) {
		mm.kind = Model;
		return mm;
	}

	const string& member = rest[0];
	if (member == "input_count" || member == "output_count" || member == "invoke") {
		if (rest.size() > 1)
			return fail(string_format("%s.%s has no member %s.", stem.c_str(),
				member.c_str(), rest[1].c_str()));
		mm.kind = member == "invoke" ? Invoke : Property;
		mm.property = member;
		return mm;
	}

	static const regex tensorName("(input|output)(0|[1-9][0-9]{0,8})");
	smatch match;
	if (!regex_match(member, match, tensorName))
		return fail(string_format("Model %s has no member %s. Its members are its tensors "
			"(%s, %s), input_count, output_count and invoke().", stem.c_str(), member.c_str(),
			tensorList(model.inputs, "input").c_str(), tensorList(model.outputs, "output").c_str()));

	mm.tensor.output = match[1] == "output";
	mm.tensor.index = stoul(match[2]);
	const vector<TfliteTensor>& tensors = mm.tensor.output ? model.outputs : model.inputs;
	const char *role = mm.tensor.output ? "output" : "input";
	if (mm.tensor.index >= tensors.size())
		return fail(string_format("Model %s has %d %s tensor%s (%s): %s does not exist. "
			"Tensors are numbered from 0.", stem.c_str(), (int)tensors.size(), role,
			tensors.size() == 1 ? "" : "s", tensorList(tensors, role).c_str(), member.c_str()));

	if (rest.size() == 1) {
		mm.kind = Tensor;
		return mm;
	}

	const TfliteTensor& t = mm.tensor.info();
	const string& sub = rest[1];
	if (rest.size() > 2)
		return fail(string_format("%s.%s.%s has no member %s.", stem.c_str(), member.c_str(),
			sub.c_str(), rest[2].c_str()));

	if (sub == "raw") {
		if (t.type != TfliteType::Int8 && t.type != TfliteType::Int16) {
			string hint = t.type == TfliteType::Float32 ?
				string_format(" For a float32 tensor, the default view %s.%s already holds "
					"its values.", stem.c_str(), member.c_str()) :
				string(" The raw view exists for int8 and int16 tensors.");
			return fail(string_format("Tensor %s of model %s is %s: it has no raw view.%s",
				member.c_str(), stem.c_str(), tfliteTypeName(t.type), hint.c_str()));
		}
		mm.kind = Tensor;
		mm.tensor.raw = true;
		return mm;
	}

	if (sub == "size" || sub == "scale" || sub == "zero_point") {
		if (sub != "size" && !t.quantized)
			return fail(string_format("Tensor %s of model %s is not quantized: it has no %s.",
				member.c_str(), stem.c_str(), sub.c_str()));
		mm.kind = Property;
		mm.ofTensor = true;
		mm.property = sub;
		return mm;
	}

	return fail(string_format("Tensor %s of model %s has no member %s. Its members are "
		"size, scale, zero_point and raw.", member.c_str(), stem.c_str(), sub.c_str()));
}

void ModelMember::report(SourceLocation *loc) const {
	if (!error.empty())
		yyerrorcpp(error, loc);
}

Node *ModelMember::constant(location_t loc) const {
	const TfliteModel& model = tensor.model->getModel();
	if (property == "input_count")
		return getNodeForUIntConst(model.inputs.size(), loc);
	if (property == "output_count")
		return getNodeForUIntConst(model.outputs.size(), loc);

	const TfliteTensor& t = tensor.info();
	if (property == "size")
		return getNodeForUIntConst(t.elements, loc);
	if (property == "scale")
		return new Float(t.scale, loc);
	return getNodeForIntConst(t.zeroPoint, loc);
}
