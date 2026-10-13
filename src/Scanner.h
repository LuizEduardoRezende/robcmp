
#pragma once

#include <string>
#include <vector>
#include <filesystem>

#include "SourceLocation.h"

using namespace std;

static const char *file_not_found = "No such file or directory '%s'.";

typedef void* yyscan_t;
extern int errorsfound;
extern vector<filesystem::path> includeDirs;
// The -I dirs alone, in the order given. includeDirs mixes them with the
// dirs of the source files and is searched backwards.
extern vector<filesystem::path> cmdlineIncludeDirs;

void yyerror(location_t *loc, yyscan_t scanner, const char *s);
void USEerror(location_t *loc, yyscan_t scanner, const char *s);
void MAINerror(location_t *loc, yyscan_t scanner, const char *s);
void yyerrorcpp(const string& s, SourceLocation *n, bool semantic);
void yywarncpp(const string& s, SourceLocation *n);

bool parseFile(const string &source);
bool parseUseFile(const string& use, location_t loc, bool isPath = false);
