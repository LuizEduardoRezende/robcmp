#pragma once

/*
 * Values known at compile time, read from the AST of the program: the arena
 * of ai_model(...), the index of a tensor element, the size of a vector.
 */

#include <cstdint>

class Node;

// Value of an integer constant: a literal, a const scalar or named constant
// whose value is one, or +, - and * over them. False for anything else.
bool integerConstant(Node *n, int64_t& value);

// Number of elements of a vector whose size is known at compile time: an
// array literal, a variable that holds one, a tensor read whole. False for a
// vector of unknown size, such as a parameter float[].
bool staticSize(Node *vector, int64_t& size);
