#include <unordered_map>
#include <experimental/optional>
#include "dsl/ast.h"

#pragma once

namespace dsl {
    enum class Type {
        Integer, List, PredicateLambda, OneArgumentLambda, TwoArgumentsLambda, Null
    };

    struct Signature {
        Type return_type;
        std::vector<Type> arguments_type;
    };

    using TypeEnvironment = std::unordered_map<Variable, Type>;

    Signature get_signature(Function function);
    std::experimental::optional<TypeEnvironment> check(const Statement& statement, const TypeEnvironment &env);
    std::experimental::optional<TypeEnvironment> generate_type_environment(const Program &program);
    bool is_valid(const Program& program);
}