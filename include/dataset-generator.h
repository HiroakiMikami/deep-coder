#pragma once

#include <experimental/optional>

struct IntegerConstraint {
    std::experimental::optional<int> min;
    std::experimental::optional<int> max;
    std::experimental::optional<int> sign;
    std::experimental::optional<bool> is_even;

    IntegerConstraint();
};

std::experimental::optional<int> generate_integer(const IntegerConstraint& constraint);