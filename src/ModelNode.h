#include "Node.h"

class ModelNode : public Node {
public:
    std::string name;        // model variable name
    std::string member;      // member name (e.g., "input", "output", "invoke")
    ParamsCall *params;      // constructor/call params
    Node *value;             // assigned value (for assignments)

    // Constructor for model instantiation
    ModelNode(const char *n, ParamsCall *p, location_t l)
        : Node(l), name(n), params(p), value(nullptr) {}

    // Constructor for member access/assignment
    ModelNode(const char *n, const char *m, Node *v, location_t l)
        : Node(l), name(n), member(m), params(nullptr), value(v) {}
};

