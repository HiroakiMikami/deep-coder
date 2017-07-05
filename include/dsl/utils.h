#include <iostream>
#include "dsl/ast.h"
#include "dsl/interpreter.h"

#pragma once

namespace dsl {
    std::string stringify(const dsl::Function &func);
    std::string stringify(const dsl::OneArgumentLambda &lambda);
    std::string stringify(const dsl::TwoArgumentsLambda &lambda);
    std::string stringify(const dsl::PredicateLambda &pred);
    std::ostream &operator<<(std::ostream &stream, const Argument &argument);
    std::ostream &operator<<(std::ostream &stream, const Statement &statement);
    std::ostream &operator<<(std::ostream &stream, const Program &program);

    std::ostream &operator<<(std::ostream &stream, const Value &value);
    std::ostream &operator<<(std::ostream &stream, const Input &input);
}