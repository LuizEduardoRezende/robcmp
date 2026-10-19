#include "tinyml/TfliteModel.h"

// The only translation unit of the compiler that sees TFLM and FlatBuffers.
#include "tensorflow/lite/schema/schema_generated.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <cctype>
#include <map>
#include <set>

/*
 * Kernel -> wrapper function. One entry per tflm_add_* function declared in
 * wrappers/tflm/tflm_wrapper.h; `make -C test/tflm check` verifies that both
 * lists are the same. A kernel without an entry is unsupported.
 *
 * Builtins are keyed by the enum constant, so a misspelled op does not
 * compile. Customs are keyed by the custom_code string the TFLM resolver
 * registers them under (micro_mutable_op_resolver.h, AddCustom).
 */
struct BuiltinEntry {
	tflite::BuiltinOperator op;
	const char *function;
};

static const BuiltinEntry builtinKernels[] = {
	{tflite::BuiltinOperator_ADD, "tflm_add_add"},
	{tflite::BuiltinOperator_AVERAGE_POOL_2D, "tflm_add_average_pool_2d"},
	{tflite::BuiltinOperator_CONCATENATION, "tflm_add_concatenation"},
	{tflite::BuiltinOperator_CONV_2D, "tflm_add_conv_2d"},
	{tflite::BuiltinOperator_DEPTHWISE_CONV_2D, "tflm_add_depthwise_conv_2d"},
	{tflite::BuiltinOperator_DEPTH_TO_SPACE, "tflm_add_depth_to_space"},
	{tflite::BuiltinOperator_DEQUANTIZE, "tflm_add_dequantize"},
	{tflite::BuiltinOperator_EMBEDDING_LOOKUP, "tflm_add_embedding_lookup"},
	{tflite::BuiltinOperator_FLOOR, "tflm_add_floor"},
	{tflite::BuiltinOperator_FULLY_CONNECTED, "tflm_add_fully_connected"},
	{tflite::BuiltinOperator_L2_NORMALIZATION, "tflm_add_l2_normalization"},
	{tflite::BuiltinOperator_L2_POOL_2D, "tflm_add_l2_pool_2d"},
	{tflite::BuiltinOperator_LOGISTIC, "tflm_add_logistic"},
	{tflite::BuiltinOperator_MAX_POOL_2D, "tflm_add_max_pool_2d"},
	{tflite::BuiltinOperator_MUL, "tflm_add_mul"},
	{tflite::BuiltinOperator_RELU, "tflm_add_relu"},
	{tflite::BuiltinOperator_RELU6, "tflm_add_relu6"},
	{tflite::BuiltinOperator_RESHAPE, "tflm_add_reshape"},
	{tflite::BuiltinOperator_RESIZE_BILINEAR, "tflm_add_resize_bilinear"},
	{tflite::BuiltinOperator_SOFTMAX, "tflm_add_softmax"},
	{tflite::BuiltinOperator_SPACE_TO_DEPTH, "tflm_add_space_to_depth"},
	{tflite::BuiltinOperator_SVDF, "tflm_add_svdf"},
	{tflite::BuiltinOperator_TANH, "tflm_add_tanh"},
	{tflite::BuiltinOperator_PAD, "tflm_add_pad"},
	{tflite::BuiltinOperator_GATHER, "tflm_add_gather"},
	{tflite::BuiltinOperator_BATCH_TO_SPACE_ND, "tflm_add_batch_to_space_nd"},
	{tflite::BuiltinOperator_SPACE_TO_BATCH_ND, "tflm_add_space_to_batch_nd"},
	{tflite::BuiltinOperator_TRANSPOSE, "tflm_add_transpose"},
	{tflite::BuiltinOperator_MEAN, "tflm_add_mean"},
	{tflite::BuiltinOperator_SUB, "tflm_add_sub"},
	{tflite::BuiltinOperator_DIV, "tflm_add_div"},
	{tflite::BuiltinOperator_SQUEEZE, "tflm_add_squeeze"},
	{tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM, "tflm_add_unidirectional_sequence_lstm"},
	{tflite::BuiltinOperator_STRIDED_SLICE, "tflm_add_strided_slice"},
	{tflite::BuiltinOperator_EXP, "tflm_add_exp"},
	{tflite::BuiltinOperator_SPLIT, "tflm_add_split"},
	{tflite::BuiltinOperator_LOG_SOFTMAX, "tflm_add_log_softmax"},
	{tflite::BuiltinOperator_CAST, "tflm_add_cast"},
	{tflite::BuiltinOperator_PRELU, "tflm_add_prelu"},
	{tflite::BuiltinOperator_MAXIMUM, "tflm_add_maximum"},
	{tflite::BuiltinOperator_ARG_MAX, "tflm_add_arg_max"},
	{tflite::BuiltinOperator_MINIMUM, "tflm_add_minimum"},
	{tflite::BuiltinOperator_LESS, "tflm_add_less"},
	{tflite::BuiltinOperator_NEG, "tflm_add_neg"},
	{tflite::BuiltinOperator_PADV2, "tflm_add_padv2"},
	{tflite::BuiltinOperator_GREATER, "tflm_add_greater"},
	{tflite::BuiltinOperator_GREATER_EQUAL, "tflm_add_greater_equal"},
	{tflite::BuiltinOperator_LESS_EQUAL, "tflm_add_less_equal"},
	{tflite::BuiltinOperator_SLICE, "tflm_add_slice"},
	{tflite::BuiltinOperator_SIN, "tflm_add_sin"},
	{tflite::BuiltinOperator_TRANSPOSE_CONV, "tflm_add_transpose_conv"},
	{tflite::BuiltinOperator_EXPAND_DIMS, "tflm_add_expand_dims"},
	{tflite::BuiltinOperator_EQUAL, "tflm_add_equal"},
	{tflite::BuiltinOperator_NOT_EQUAL, "tflm_add_not_equal"},
	{tflite::BuiltinOperator_LOG, "tflm_add_log"},
	{tflite::BuiltinOperator_SUM, "tflm_add_sum"},
	{tflite::BuiltinOperator_SQRT, "tflm_add_sqrt"},
	{tflite::BuiltinOperator_RSQRT, "tflm_add_rsqrt"},
	{tflite::BuiltinOperator_SHAPE, "tflm_add_shape"},
	{tflite::BuiltinOperator_ARG_MIN, "tflm_add_arg_min"},
	{tflite::BuiltinOperator_REDUCE_MAX, "tflm_add_reduce_max"},
	{tflite::BuiltinOperator_PACK, "tflm_add_pack"},
	{tflite::BuiltinOperator_LOGICAL_OR, "tflm_add_logical_or"},
	{tflite::BuiltinOperator_LOGICAL_AND, "tflm_add_logical_and"},
	{tflite::BuiltinOperator_LOGICAL_NOT, "tflm_add_logical_not"},
	{tflite::BuiltinOperator_UNPACK, "tflm_add_unpack"},
	{tflite::BuiltinOperator_REDUCE_MIN, "tflm_add_reduce_min"},
	{tflite::BuiltinOperator_FLOOR_DIV, "tflm_add_floor_div"},
	{tflite::BuiltinOperator_SQUARE, "tflm_add_square"},
	{tflite::BuiltinOperator_ZEROS_LIKE, "tflm_add_zeros_like"},
	{tflite::BuiltinOperator_FILL, "tflm_add_fill"},
	{tflite::BuiltinOperator_FLOOR_MOD, "tflm_add_floor_mod"},
	{tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR, "tflm_add_resize_nearest_neighbor"},
	{tflite::BuiltinOperator_LEAKY_RELU, "tflm_add_leaky_relu"},
	{tflite::BuiltinOperator_SQUARED_DIFFERENCE, "tflm_add_squared_difference"},
	{tflite::BuiltinOperator_MIRROR_PAD, "tflm_add_mirror_pad"},
	{tflite::BuiltinOperator_ABS, "tflm_add_abs"},
	{tflite::BuiltinOperator_SPLIT_V, "tflm_add_split_v"},
	{tflite::BuiltinOperator_CEIL, "tflm_add_ceil"},
	{tflite::BuiltinOperator_REVERSE_V2, "tflm_add_reverse_v2"},
	{tflite::BuiltinOperator_ADD_N, "tflm_add_add_n"},
	{tflite::BuiltinOperator_GATHER_ND, "tflm_add_gather_nd"},
	{tflite::BuiltinOperator_COS, "tflm_add_cos"},
	{tflite::BuiltinOperator_ELU, "tflm_add_elu"},
	{tflite::BuiltinOperator_QUANTIZE, "tflm_add_quantize"},
	{tflite::BuiltinOperator_ROUND, "tflm_add_round"},
	{tflite::BuiltinOperator_HARD_SWISH, "tflm_add_hard_swish"},
	{tflite::BuiltinOperator_IF, "tflm_add_if"},
	{tflite::BuiltinOperator_WHILE, "tflm_add_while"},
	{tflite::BuiltinOperator_SELECT_V2, "tflm_add_select_v2"},
	{tflite::BuiltinOperator_BATCH_MATMUL, "tflm_add_batch_matmul"},
	{tflite::BuiltinOperator_CUMSUM, "tflm_add_cumsum"},
	{tflite::BuiltinOperator_CALL_ONCE, "tflm_add_call_once"},
	{tflite::BuiltinOperator_BROADCAST_TO, "tflm_add_broadcast_to"},
	{tflite::BuiltinOperator_VAR_HANDLE, "tflm_add_var_handle"},
	{tflite::BuiltinOperator_READ_VARIABLE, "tflm_add_read_variable"},
	{tflite::BuiltinOperator_ASSIGN_VARIABLE, "tflm_add_assign_variable"},
	{tflite::BuiltinOperator_BROADCAST_ARGS, "tflm_add_broadcast_args"},
	{tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE, "tflm_add_dynamic_update_slice"},
	{tflite::BuiltinOperator_REDUCE_ALL, "tflm_add_reduce_all"},
};

struct CustomEntry {
	const char *op;
	const char *function;
};

static const CustomEntry customKernels[] = {
	// Signal processing, the audio pipeline of Micro Speech
	{"SignalDelay", "tflm_add_delay"},
	{"SignalEnergy", "tflm_add_energy"},
	{"SignalFftAutoScale", "tflm_add_fft_auto_scale"},
	{"SignalFilterBank", "tflm_add_filter_bank"},
	{"SignalFilterBankLog", "tflm_add_filter_bank_log"},
	{"SignalFilterBankSpectralSubtraction", "tflm_add_filter_bank_spectral_subtraction"},
	{"SignalFilterBankSquareRoot", "tflm_add_filter_bank_square_root"},
	{"SignalFramer", "tflm_add_framer"},
	{"SignalIrfft", "tflm_add_irfft"},
	{"SignalOverlapAdd", "tflm_add_overlap_add"},
	{"SignalPCAN", "tflm_add_pcan"},
	{"SignalRfft", "tflm_add_rfft"},
	{"SignalStacker", "tflm_add_stacker"},
	{"SignalWindow", "tflm_add_window"},

	// Others
	{"BasicClassifier", "tflm_add_basic_classifier"},
	{"CIRCULAR_BUFFER", "tflm_add_circular_buffer"},
	{"TFLM_DECODE", "tflm_add_decode"},
	{"TFLite_Detection_PostProcess", "tflm_add_detection_postprocess"},
	{"ethos-u", "tflm_add_ethos_u"},	// GetString_ETHOSU()
};

const char *tfliteTypeName(TfliteType t) {
	switch (t) {
		case TfliteType::Float32: return "float32";
		case TfliteType::Float16: return "float16";
		case TfliteType::Int32:   return "int32";
		case TfliteType::UInt8:   return "uint8";
		case TfliteType::Int64:   return "int64";
		case TfliteType::String:  return "string";
		case TfliteType::Bool:    return "bool";
		case TfliteType::Int16:   return "int16";
		case TfliteType::Int8:    return "int8";
		case TfliteType::UInt16:  return "uint16";
		case TfliteType::UInt32:  return "uint32";
		default:                  return "other";
	}
}

static TfliteType convertType(tflite::TensorType t) {
	switch (t) {
		case tflite::TensorType_FLOAT32: return TfliteType::Float32;
		case tflite::TensorType_FLOAT16: return TfliteType::Float16;
		case tflite::TensorType_INT32:   return TfliteType::Int32;
		case tflite::TensorType_UINT8:   return TfliteType::UInt8;
		case tflite::TensorType_INT64:   return TfliteType::Int64;
		case tflite::TensorType_STRING:  return TfliteType::String;
		case tflite::TensorType_BOOL:    return TfliteType::Bool;
		case tflite::TensorType_INT16:   return TfliteType::Int16;
		case tflite::TensorType_INT8:    return TfliteType::Int8;
		case tflite::TensorType_UINT16:  return TfliteType::UInt16;
		case tflite::TensorType_UINT32:  return TfliteType::UInt32;
		default:                         return TfliteType::Other;
	}
}

/*
 * Old converters leave builtin_code zeroed (= ADD) and store the real code in
 * deprecated_builtin_code; new ones store codes above 127 only in
 * builtin_code. Reading builtin_code alone silently reports ADD for every op
 * of an old model (spam.tflite). Same rule as tflite::GetBuiltinCode, in
 * schema_utils.cc, which is not compiled into the compiler for one line.
 */
static tflite::BuiltinOperator builtinCode(const tflite::OperatorCode *oc) {
	return std::max(oc->builtin_code(),
		static_cast<tflite::BuiltinOperator>(oc->deprecated_builtin_code()));
}

static void fail(std::string &error, const std::string &path, const std::string &msg) {
	error = "model '" + path + "': " + msg;
}

static bool readTensor(const tflite::SubGraph *sg, int32_t index, const char *role,
	unsigned position, const std::string &path, TfliteTensor &out, std::string &error) {

	const auto *tensors = sg->tensors();
	if (!tensors || index < 0 || (uint32_t)index >= tensors->size()) {
		fail(error, path, std::string(role) + " " + std::to_string(position) +
			" refers to tensor " + std::to_string(index) + ", which does not exist");
		return false;
	}

	const tflite::Tensor *t = tensors->Get(index);
	out.name = t->name() ? t->name()->str() : "";
	out.type = convertType(t->type());

	out.elements = 1;
	if (t->shape()) {
		for (int32_t dim : *t->shape()) {
			if (dim < 0) {
				fail(error, path, std::string(role) + " " + std::to_string(position) +
					" has a dynamic shape, which TFLM does not support");
				return false;
			}
			out.shape.push_back(dim);
			out.elements *= (uint64_t)dim;
		}
	}

	const tflite::QuantizationParameters *q = t->quantization();
	if (q && q->scale() && q->scale()->size() > 0) {
		out.quantized = true;
		out.scale = q->scale()->Get(0);
		out.quantChannels = q->scale()->size();
		if (q->zero_point() && q->zero_point()->size() > 0)
			out.zeroPoint = q->zero_point()->Get(0);
	}
	return true;
}

static std::string typeName(tflite::TensorType t) {
	std::string n = tflite::EnumNameTensorType(t);
	for (char &c : n)
		c = (char)std::tolower((unsigned char)c);
	return n.empty() ? "type " + std::to_string((int)t) : n;
}

/*
 * The combinations of input and weight types that TFLM runs, as its kernels
 * check them in Prepare (kernels/conv_common.cc, depthwise_conv_common.cc,
 * transpose_conv.cc, fully_connected.cc). Float input with integer weights --
 * the "hybrid" or "dynamic range" quantization of TFLite -- is refused, and
 * the model fails when the interpreter is created.
 */
static bool tflmRunsTypes(tflite::BuiltinOperator op, tflite::TensorType in,
	tflite::TensorType w) {
	using namespace tflite;
	bool intWeights = w == TensorType_INT8 || w == TensorType_INT4;
	switch (op) {
		case BuiltinOperator_CONV_2D:
			return (in == TensorType_FLOAT32 && w == TensorType_FLOAT32) ||
				(in == TensorType_INT16 && w == TensorType_INT8) ||
				(in == TensorType_INT8 && intWeights);
		case BuiltinOperator_DEPTHWISE_CONV_2D:
			return in == w || (in == TensorType_INT8 && intWeights) ||
				(in == TensorType_INT16 && w == TensorType_INT8);
		case BuiltinOperator_TRANSPOSE_CONV:
			return in == w || (in == TensorType_INT16 && w == TensorType_INT8);
		case BuiltinOperator_FULLY_CONNECTED:
			return !((in == TensorType_FLOAT32 && w != TensorType_FLOAT32) ||
				(in == TensorType_INT8 && !intWeights) ||
				(in == TensorType_INT16 && w != TensorType_INT8));
		default:
			return true;
	}
}

static const tflite::Tensor *operand(const tflite::SubGraph *sg,
	const flatbuffers::Vector<int32_t> *list, unsigned position) {
	if (!list || position >= list->size() || !sg->tensors())
		return nullptr;
	int32_t index = list->Get(position);
	if (index < 0 || (uint32_t)index >= sg->tensors()->size())
		return nullptr;
	return sg->tensors()->Get(index);
}

// An operation TFLM refuses for its tensor types, or "".
static std::string typeIncompatibility(const tflite::SubGraph *sg, const tflite::Operator *op,
	tflite::BuiltinOperator code) {
	// TRANSPOSE_CONV takes (output shape, weights, input); the others take
	// (input, weights).
	unsigned inputPosition = code == tflite::BuiltinOperator_TRANSPOSE_CONV ? 2 : 0;
	const tflite::Tensor *in = operand(sg, op->inputs(), inputPosition);
	const tflite::Tensor *w = operand(sg, op->inputs(), 1);
	if (!in || !w || tflmRunsTypes(code, in->type(), w->type()))
		return "";
	return std::string(tflite::EnumNameBuiltinOperator(code)) + " with " +
		typeName(in->type()) + " input and " + typeName(w->type()) + " weights";
}

/*
 * TFLM gives each tensor the size of its shape in the file: it does not resize
 * tensors at run time. TFLite_Detection_PostProcess writes up to
 * max_detections detections into its outputs, so outputs with no shape -- as
 * models converted for TFLite, which resizes them, have -- would be written
 * past their end.
 */
static std::string shapeIncompatibility(const tflite::SubGraph *sg, const tflite::Operator *op,
	const std::string &customCode) {
	if (customCode != "TFLite_Detection_PostProcess" || !op->outputs())
		return "";
	std::string unshaped;
	for (unsigned i = 0; i < op->outputs()->size(); i++) {
		const tflite::Tensor *t = operand(sg, op->outputs(), i);
		if (t && (!t->shape() || t->shape()->size() == 0))
			unshaped += (unshaped.empty() ? "" : ", ") + std::to_string(i);
	}
	if (unshaped.empty())
		return "";
	return "outputs " + unshaped + " of TFLite_Detection_PostProcess with no shape in the "
		"file (TFLM sizes tensors by the file, and the operator would write past their end)";
}

static std::string wrapperFunctionFor(tflite::BuiltinOperator op) {
	for (const BuiltinEntry &e : builtinKernels)
		if (e.op == op)
			return e.function;
	return "";
}

static std::string wrapperFunctionFor(const std::string &customCode) {
	for (const CustomEntry &e : customKernels)
		if (customCode == e.op)
			return e.function;
	return "";
}

bool TfliteModel::load(const std::string &path, TfliteModel &model, std::string &error) {
	model = TfliteModel();
	model.path = path;

	std::ifstream file(path, std::ios::binary);
	if (!file) {
		fail(error, path, "file not found or not readable");
		return false;
	}
	model.bytes.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());

	// "TFL3" at bytes 4..7. Checked apart from the verifier to tell "not a
	// model at all" from "a damaged model".
	const uint8_t *data = model.bytes.data();
	if (model.bytes.size() < 8 || !tflite::ModelBufferHasIdentifier(data)) {
		fail(error, path, "not a TFLite model (TFL3 identifier missing)");
		return false;
	}

	// Structural verification: every offset and vector length is checked to
	// stay inside the buffer. Without it, a truncated file makes the accessors
	// below read out of bounds -- inside the compiler.
	flatbuffers::Verifier verifier(data, model.bytes.size());
	if (!tflite::VerifyModelBuffer(verifier)) {
		fail(error, path, "corrupted or truncated TFLite file (FlatBuffers verification failed)");
		return false;
	}

	const tflite::Model *m = tflite::GetModel(data);
	model.schemaVersion = m->version();

	if (!m->subgraphs() || m->subgraphs()->size() == 0) {
		fail(error, path, "the model has no subgraph");
		return false;
	}
	model.subgraphs = m->subgraphs()->size();

	// Interface: subgraph 0 is the main model; the others are bodies of
	// IF/WHILE/CALL_ONCE, run internally by TFLM.
	const tflite::SubGraph *main = m->subgraphs()->Get(0);
	if (main->inputs()) {
		for (unsigned i = 0; i < main->inputs()->size(); i++) {
			TfliteTensor t;
			if (!readTensor(main, main->inputs()->Get(i), "input", i, path, t, error))
				return false;
			model.inputs.push_back(t);
		}
	}
	if (main->outputs()) {
		for (unsigned i = 0; i < main->outputs()->size(); i++) {
			TfliteTensor t;
			if (!readTensor(main, main->outputs()->Get(i), "output", i, path, t, error))
				return false;
			model.outputs.push_back(t);
		}
	}

	// Kernels: those referenced by an operator of any subgraph. TFLM resolves
	// kernels per operator, so an operator code no operator uses needs no
	// registration -- and must not be reported as unsupported.
	const auto *codes = m->operator_codes();
	std::map<std::string, TfliteKernel> kernels;
	std::set<std::string> incompatibilities;
	for (const tflite::SubGraph *sg : *m->subgraphs()) {
		if (!sg->operators())
			continue;
		for (const tflite::Operator *op : *sg->operators()) {
			uint32_t idx = op->opcode_index();
			if (!codes || idx >= codes->size()) {
				fail(error, path, "operator refers to operator code " +
					std::to_string(idx) + ", which does not exist");
				return false;
			}

			const tflite::OperatorCode *oc = codes->Get(idx);
			tflite::BuiltinOperator code = builtinCode(oc);
			TfliteKernel k;
			if (code == tflite::BuiltinOperator_CUSTOM) {
				if (!oc->custom_code() || oc->custom_code()->size() == 0) {
					fail(error, path, "custom operator without a name");
					return false;
				}
				k.op = oc->custom_code()->str();
				k.custom = true;
				k.wrapperFunction = wrapperFunctionFor(k.op);
				std::string why = shapeIncompatibility(sg, op, k.op);
				if (!why.empty())
					incompatibilities.insert(why);
			} else {
				const char *name = tflite::EnumNameBuiltinOperator(code);
				k.op = (name && *name) ? name : "BUILTIN_" + std::to_string((int)code);
				k.wrapperFunction = wrapperFunctionFor(code);
				std::string why = typeIncompatibility(sg, op, code);
				if (!why.empty())
					incompatibilities.insert(why);
			}
			kernels.emplace(k.op, k);
		}
	}
	for (auto &kv : kernels)
		model.kernels.push_back(kv.second);
	model.incompatibilities.assign(incompatibilities.begin(), incompatibilities.end());

	return true;
}

std::vector<TfliteKernel> TfliteModel::supportedKernels() {
	std::vector<TfliteKernel> result;
	for (const BuiltinEntry &e : builtinKernels)
		result.push_back({tflite::EnumNameBuiltinOperator(e.op), false, e.function});
	for (const CustomEntry &e : customKernels)
		result.push_back({e.op, true, e.function});
	return result;
}
