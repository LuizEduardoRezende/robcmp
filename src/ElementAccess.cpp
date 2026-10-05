#include "ElementAccess.h"

ElementAccess::ElementAccess(const string &i, Node *pos, location_t loc):
	BaseArrayOper(i, pos, NULL, loc) {
	addChild(pos);
}

string ElementAccess::describe() const {
	return getIdent() + "[...]";
}

UpdateElementArray::UpdateElementArray(const string &i, Node *pos, ArrayElements *elements, location_t loc):
	ElementAccess(i, pos, loc), elements(elements) {
}

LoadElementMember::LoadElementMember(const string &i, Node *pos, const string &member, location_t loc):
	ElementAccess(i, pos, loc), member(member) {
}

string LoadElementMember::describe() const {
	return ElementAccess::describe() + "." + member;
}

UpdateElementMember::UpdateElementMember(const string &i, Node *pos, const string &member, Node *expr, location_t loc):
	ElementAccess(i, pos, loc), member(member), expr(expr) {
	addChild(expr);
}

UpdateElementMember::UpdateElementMember(const string &i, Node *pos, const string &member, ArrayElements *elements, location_t loc):
	ElementAccess(i, pos, loc), member(member), elements(elements) {
}

string UpdateElementMember::describe() const {
	return ElementAccess::describe() + "." + member;
}
