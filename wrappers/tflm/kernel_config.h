#pragma once

#define ENABLE_ADD 0
#define ENABLE_AVERAGE_POOL_2D 1
#define ENABLE_CONCATENATION 0
#define ENABLE_CONV_2D 1
#define ENABLE_DEPTHWISE_CONV_2D 1
#define ENABLE_DEPTH_TO_SPACE 0
#define ENABLE_DEQUANTIZE 0
#define ENABLE_EMBEDDING_LOOKUP 0
#define ENABLE_FLOOR 0
#define ENABLE_FULLY_CONNECTED 1
#define ENABLE_HASHTABLE_LOOKUP 0 //Não suportado no TFLM
#define ENABLE_L2_NORMALIZATION 0
#define ENABLE_L2_POOL_2D 0
#define ENABLE_LOCAL_RESPONSE_NORMALIZATION 0 //Não suportado no TFLM
#define ENABLE_LOGISTIC 0
#define ENABLE_LSH_PROJECTION 0 //Não suportado no TFLM
#define ENABLE_LSTM 0 //Não suportado no TFLM
#define ENABLE_MAX_POOL_2D 0
#define ENABLE_MUL 0
#define ENABLE_RELU 0
#define ENABLE_RELU_N1_TO_1 0 //Não suportado no TFLM
#define ENABLE_RELU6 0
#define ENABLE_RESHAPE 1
#define ENABLE_RESIZE_BILINEAR 0
#define ENABLE_RNN 0 //Não suportado no TFLM
#define ENABLE_SOFTMAX 1
#define ENABLE_SPACE_TO_DEPTH 0
#define ENABLE_SVDF 0
#define ENABLE_TANH 0
#define ENABLE_CONCAT_EMBEDDINGS 0 //Não suportado no TFLM
#define ENABLE_SKIP_GRAM 0 //Não suportado no TFLM
#define ENABLE_CALL 0 //Não suportado no TFLM
#define ENABLE_CUSTOM 0 //Não suportado no TFLM
#define ENABLE_EMBEDDING_LOOKUP_SPARSE 0 //Não suportado no TFLM
#define ENABLE_PAD 0
#define ENABLE_UNIDIRECTIONAL_SEQUENCE_RNN 0 //Não suportado no TFLM
#define ENABLE_GATHER 1
#define ENABLE_BATCH_TO_SPACE_ND 0
#define ENABLE_SPACE_TO_BATCH_ND 0
#define ENABLE_TRANSPOSE 0
#define ENABLE_MEAN 1
#define ENABLE_SUB 0
#define ENABLE_DIV 0
#define ENABLE_SQUEEZE 0
#define ENABLE_UNIDIRECTIONAL_SEQUENCE_LSTM 0
#define ENABLE_STRIDED_SLICE 0
#define ENABLE_BIDIRECTIONAL_SEQUENCE_RNN 0 //Não suportado no TFLM
#define ENABLE_EXP 0
#define ENABLE_TOPK_V2 0 //Não suportado no TFLM
#define ENABLE_SPLIT 0
#define ENABLE_LOG_SOFTMAX 0
#define ENABLE_DELEGATE 0 //Não suportado no TFLM
#define ENABLE_BIDIRECTIONAL_SEQUENCE_LSTM 0 //Não suportado no TFLM
#define ENABLE_CAST 0
#define ENABLE_PRELU 0
#define ENABLE_MAXIMUM 0
#define ENABLE_ARG_MAX 0
#define ENABLE_MINIMUM 0
#define ENABLE_LESS 0
#define ENABLE_NEG 0
#define ENABLE_PADV2 0
#define ENABLE_GREATER 0
#define ENABLE_GREATER_EQUAL 0
#define ENABLE_LESS_EQUAL 0
#define ENABLE_SELECT 0 //Não suportado no TFLM
#define ENABLE_SLICE 0
#define ENABLE_SIN 0
#define ENABLE_TRANSPOSE_CONV 0
#define ENABLE_SPARSE_TO_DENSE 0 //Não suportado no TFLM
#define ENABLE_TILE 0 //Não suportado no TFLM
#define ENABLE_EXPAND_DIMS 0
#define ENABLE_EQUAL 0
#define ENABLE_NOT_EQUAL 0
#define ENABLE_LOG 0
#define ENABLE_SUM 0
#define ENABLE_SQRT 0
#define ENABLE_RSQRT 0
#define ENABLE_SHAPE 0
#define ENABLE_POW 0 //Não suportado no TFLM
#define ENABLE_ARG_MIN 0
#define ENABLE_FAKE_QUANT 0 //Não suportado no TFLM
#define ENABLE_REDUCE_PROD 0 //Não suportado no TFLM
#define ENABLE_REDUCE_MAX 0
#define ENABLE_PACK 0
#define ENABLE_LOGICAL_OR 0
#define ENABLE_ONE_HOT 0 //Não suportado no TFLM
#define ENABLE_LOGICAL_AND 0
#define ENABLE_LOGICAL_NOT 0
#define ENABLE_UNPACK 0
#define ENABLE_REDUCE_MIN 0
#define ENABLE_FLOOR_DIV 0
#define ENABLE_REDUCE_ANY 0 //Não suportado no TFLM
#define ENABLE_SQUARE 0
#define ENABLE_ZEROS_LIKE 0
#define ENABLE_FILL 0
#define ENABLE_FLOOR_MOD 0
#define ENABLE_RANGE 0 //Não suportado no TFLM
#define ENABLE_RESIZE_NEAREST_NEIGHBOR 0
#define ENABLE_LEAKY_RELU 0
#define ENABLE_SQUARED_DIFFERENCE 0
#define ENABLE_MIRROR_PAD 0
#define ENABLE_ABS 0
#define ENABLE_SPLIT_V 0
#define ENABLE_UNIQUE 0 //Não suportado no TFLM
#define ENABLE_CEIL 0
#define ENABLE_REVERSE_V2 0
#define ENABLE_ADD_N 0
#define ENABLE_GATHER_ND 0
#define ENABLE_COS 0
#define ENABLE_WHERE 0 // Não suportado no TFLM
#define ENABLE_RANK 0 //Não suportado no TFLM
#define ENABLE_ELU 0
#define ENABLE_REVERSE_SEQUENCE 0 //Não suportado no TFLM
#define ENABLE_MATRIX_DIAG 0 //Não suportado no TFLM
#define ENABLE_QUANTIZE 0
#define ENABLE_MATRIX_SET_DIAG 0 //Não suportado no TFLM
#define ENABLE_ROUND 0
#define ENABLE_HARD_SWISH 0
#define ENABLE_IF 0
#define ENABLE_WHILE 0
#define ENABLE_NON_MAX_SUPPRESSION_V4 0 //Não suportado no TFLM
#define ENABLE_NON_MAX_SUPPRESSION_V5 0 //Não suportado no TFLM
#define ENABLE_SCATTER_ND 0 //Não suportado no TFLM
#define ENABLE_SELECT_V2 0
#define ENABLE_DENSIFY 0 //Não suportado no TFLM
#define ENABLE_SEGMENT_SUM 0 //Não suportado no TFLM
#define ENABLE_BATCH_MATMUL 0
#define ENABLE_PLACEHOLDER_FOR_GREATER_OP_CODES 0 //Não suportado no TFLM
#define ENABLE_CUMSUM 0
#define ENABLE_CALL_ONCE 0
#define ENABLE_BROADCAST_TO 0
#define ENABLE_RFFT2D 0 //Não suportado no TFLM
#define ENABLE_CONV_3D 0 //Não suportado no TFLM
#define ENABLE_IMAG 0 //Não suportado no TFLM
#define ENABLE_REAL 0 //Não suportado no TFLM
#define ENABLE_COMPLEX_ABS 0 //Não suportado no TFLM
#define ENABLE_HASHTABLE 0 //Não suportado no TFLM
#define ENABLE_HASHTABLE_FIND 0 //Não suportado no TFLM
#define ENABLE_HASHTABLE_IMPORT 0 //Não suportado no TFLM
#define ENABLE_HASHTABLE_SIZE 0 //Não suportado no TFLM
#define ENABLE_REDUCE_ALL 0 //Não suportado no TFLM
#define ENABLE_CONV_3D_TRANSPOSE 0 //Não suportado no TFLM
#define ENABLE_VAR_HANDLE 0
#define ENABLE_READ_VARIABLE 0
#define ENABLE_ASSIGN_VARIABLE 0
#define ENABLE_BROADCAST_ARGS 0
#define ENABLE_RANDOM_STANDARD_NORMAL 0
#define ENABLE_BUCKETIZE 0 //Não suportado no TFLM
#define ENABLE_RANDOM_UNIFORM 0 //Não suportado no TFLM
#define ENABLE_MULTINOMIAL 0 //Não suportado no TFLM
#define ENABLE_GELU 0 //Não suportado no TFLM
#define ENABLE_DYNAMIC_UPDATE_SLICE 0 //Não suportado no TFLM
#define ENABLE_RELU_0_TO_1 0 //Não suportado no TFLM
#define ENABLE_UNSORTED_SEGMENT_PROD 0 //Não suportado no TFLM
#define ENABLE_UNSORTED_SEGMENT_MAX 0 //Não suportado no TFLM
#define ENABLE_UNSORTED_SEGMENT_SUM 0 //Não suportado no TFLM
#define ENABLE_ATAN2 0 //Não suportado no TFLM
#define ENABLE_UNSORTED_SEGMENT_MIN 0 //Não suportado no TFLM
#define ENABLE_SIGN 0 //Não suportado no TFLM
#define ENABLE_BITCAST 0 //Não suportado no TFLM
#define ENABLE_BITWISE_XOR 0 //Não suportado no TFLM
#define ENABLE_RIGHT_SHIFT 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_LOGISTIC 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_ADD 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_DIVIDE 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_MULTIPLY 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_MAXIMUM 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_RESHAPE 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_CLAMP 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_CONCATENATE 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_BROADCAST_IN_DIM 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_CONVOLUTION 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_SLICE 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_CUSTOM_CALL 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_REDUCE 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_ABS 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_AND 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_COSINE 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_EXPONENTIAL 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_FLOOR 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_LOG 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_MINIMUM 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_NEGATE 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_OR 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_POWER 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_REMAINDER 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_RSQRT 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_SELECT 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_SUBTRACT 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_TANH 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_SCATTER 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_COMPARE 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_CONVERT 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_DYNAMIC_SLICE 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_DYNAMIC_UPDATE_SLICE 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_PAD 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_IOTA 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_DOT_GENERAL 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_REDUCE_WINDOW 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_SORT 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_WHILE 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_GATHER 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_TRANSPOSE 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_DILATE 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_RNG_BIT_GENERATOR 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_REDUCE_WINDOW 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_COMPOSITE 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_SHIFT_LEFT 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_CBRT 0 //Não suportado no TFLM
#define ENABLE_STABLEHLO_CASE 0 //Não suportado no TFLM

