#pragma once

/*
 * Compile-time analysis of a .tflite model.
 *
 * Runs before the AST is walked: reads the model file once and extracts what
 * the semantic analysis and the code generation need to know about it -- the
 * interface tensors (type, flattened size, quantization) and the kernels the
 * model requires, each mapped to the wrapper function that registers it.
 *
 * This header deliberately exposes no TFLM or FlatBuffers type, so that the
 * tflite namespace does not leak into the rest of the compiler.
 */

#include <cstdint>
#include <string>
#include <vector>

enum class TfliteType {
	Float32, Float16, Int32, UInt8, Int64, String, Bool, Int16, Int8,
	UInt16, UInt32, Other
};

const char *tfliteTypeName(TfliteType t);

struct TfliteTensor {
	std::string name;
	TfliteType type = TfliteType::Other;
	std::vector<int32_t> shape;
	uint64_t elements = 1;			// flattened: [1,96,96,1] -> 9216

	// Linear asymmetric quantization, real = scale * (code - zero_point).
	// quantChannels > 1 means per-channel; scale and zeroPoint are channel 0.
	bool quantized = false;
	float scale = 0.0f;
	int64_t zeroPoint = 0;
	unsigned quantChannels = 0;
};

struct TfliteKernel {
	std::string op;					// "FULLY_CONNECTED", or custom_code: "SignalRfft"
	bool custom = false;
	std::string wrapperFunction;	// "tflm_add_fully_connected"; empty = unsupported
};

struct TfliteModel {
	std::string path;
	std::vector<uint8_t> bytes;		// the whole file, to be embedded in the program
	unsigned schemaVersion = 0;
	unsigned subgraphs = 0;

	// Interface of subgraph 0, the main model (schema.fbs).
	std::vector<TfliteTensor> inputs;
	std::vector<TfliteTensor> outputs;

	// Kernels referenced by some operator of any subgraph, without duplicates,
	// sorted by op name.
	std::vector<TfliteKernel> kernels;

	// Reads and verifies the file. On failure returns false and sets error to
	// a message naming the file; a truncated or corrupted file is an error,
	// never a crash.
	static bool load(const std::string &path, TfliteModel &model, std::string &error);

	// Every kernel the wrapper can register, as (op, wrapper function).
	static std::vector<TfliteKernel> supportedKernels();
};
