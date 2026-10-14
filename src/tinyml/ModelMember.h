#pragma once

/*
 * What a dotted name means when it starts at a model.
 *
 * The members of a model are not symbols: they are read from the .tflite, and
 * resolve() interprets the name against it. Thus a misspelled or nonexistent
 * member gets an error that says what exists, instead of "symbol not found":
 *
 *     m                   the model itself, which is not a value
 *     m.input_count       property of the model
 *     m.output_count
 *     m.invoke            method of the model
 *     m.input0            tensor, default view: real values, float
 *     m.input0.raw        tensor, raw view: codes, the type of the tensor
 *     m.input0.size       property of the tensor
 *     m.input0.scale
 *     m.input0.zero_point
 *
 * The model may be reached through any stem: m, self.m, robot.m.
 */

#include "Node.h"
#include "tinyml/TfliteModel.h"

class AiModel;
class Variable;

// A tensor of a model, in one of its two views.
struct TensorRef {
	AiModel *model = nullptr;
	Variable *variable = nullptr;	// the variable that holds the model
	bool output = false;
	unsigned index = 0;
	bool raw = false;

	const TfliteTensor& info() const;
	string name() const;			// "input0", "output0.raw"

	// Type of the whole tensor and of one element, in this view: float for
	// the default view, the type of the tensor for the raw view.
	DataType elementType() const;
	DataType vectorType(SourceLocation *loc) const;
};

struct ModelMember {
	enum Kind {
		None,		// not a member of a model: the name is left to the compiler
		Model,		// the model itself
		Tensor,
		Property,
		Invoke,
		Error		// error already formatted; empty if the model is invalid
	};

	Kind kind = None;
	string name;					// the name as written, "m.input0.scale"
	TensorRef tensor;				// Model, Tensor, Property, Invoke: model and variable
	bool ofTensor = false;			// Property: of a tensor, or of the model
	string property;				// "size", "scale", "zero_point", "input_count", "output_count"
	string error;

	static ModelMember resolve(const string& name, Node *scope);

	// For Error: reports the error at loc, unless it is empty.
	void report(SourceLocation *loc) const;

	// For Property: the value, as a literal node.
	Node *constant(location_t loc) const;
};
