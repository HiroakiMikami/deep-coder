#pragma once

#include <experimental/optional>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "dsl/ast.h"
#include "dsl/interpreter.h"

enum class Sign {
    Zero, Positive, Negative
};

struct IntegerConstraint {
    std::experimental::optional<int> min;
    std::experimental::optional<int> max;
    std::experimental::optional<Sign> sign;
    std::experimental::optional<bool> is_even;

    IntegerConstraint();

    std::pair<std::experimental::optional<int>, std::experimental::optional<int>> range() const;
};

struct ListConstraint {
    std::experimental::optional<int> min_length;
    std::experimental::optional<int> max_length;
    std::experimental::optional<int> min;
    std::experimental::optional<int> max;
    std::unordered_set<std::experimental::optional<Sign>> sign;
    std::unordered_set<std::experimental::optional<bool>> is_even;

    ListConstraint();
    IntegerConstraint generate_integer_constraint() const;
    std::vector<IntegerConstraint> all_constraints() const;
};

struct Constraint {
    std::unordered_map<dsl::Variable, IntegerConstraint> integer_variables;
    std::unordered_map<dsl::Variable, ListConstraint> list_variables;

    std::vector<dsl::Variable> inputs;
};

struct Example {
    dsl::Input input;
    dsl::Output output;
};

std::experimental::optional<int> generate_integer(const IntegerConstraint& constraint);
std::experimental::optional<std::vector<int>> generate_list(const ListConstraint &constraint);

std::experimental::optional<Constraint> analyze(const dsl::Program &p);

std::experimental::optional<std::vector<Example>> generate_examples(const dsl::Program &p, size_t example_num = 5);