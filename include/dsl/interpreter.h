#include <vector>
#include <unordered_map>
#include <experimental/optional>
#include "dsl/ast.h"
#include "dsl/type.h"

#pragma once

namespace dsl {
    struct Value {
        Value();
        Value(int i);
        Value(const std::vector<int>& l);

        Type type;
        std::vector<int> result;

        std::experimental::optional<int> integer() const;
        std::experimental::optional<std::vector<int>> list() const;
        bool is_null() const;

        bool operator==(const Value& rhs) const;
        bool operator!=(const Value& rhs) const;
    };

    using Input = std::vector<Value>;
    using Output = Value;

    struct Environment {
        std::unordered_map<Variable, Value> variables;
        Input input;
        size_t offset;

        Environment(const std::unordered_map<Variable, Value> &variables, const Input& input);
    };

    std::experimental::optional<Output> eval(const Program &program, const Input &input);
    std::experimental::optional<Environment> proceed(const Statement &statement, const Environment &environment);
}