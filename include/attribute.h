#pragma once

#include <vector>
#include <unordered_map>
#include "dsl/ast.h"

struct Attribute {
    std::unordered_map<dsl::Function, double> function_presence;
    std::unordered_map<dsl::PredicateLambda, double> predicate_presence;
    std::unordered_map<dsl::OneArgumentLambda, double> one_argument_lambda_presence;
    std::unordered_map<dsl::TwoArgumentsLambda, double> two_arguments_lambda_presence;

    Attribute(const dsl::Program &program);
    Attribute(const std::vector<double> &attribute_vector);

    operator std::vector<double>() const;
};