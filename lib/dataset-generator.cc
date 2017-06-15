#include <random>
#include <algorithm>
#include "dataset-generator.h"

using namespace std;

static random_device rnd;
static mt19937 mt(rnd());

IntegerConstraint::IntegerConstraint() : min(), max(), sign(), is_even() {}

experimental::optional<int> generate_integer(const IntegerConstraint& constraint) {
    auto min = constraint.min.value_or(-256);
    auto max = constraint.max.value_or(255);

    // sign
    if (constraint.sign) {
        if (constraint.sign.value() > 0) {
            min = std::max(min, 1);
        } else if (constraint.sign.value() < 0){
            max = std::min(max, -1);
        } else {
            return 0;
        }
    }

    if (max < min) {
        return {};
    }

    uniform_int_distribution<> r(min, max);
    if (constraint.is_even) {
        if (constraint.is_even.value()) {
            // Even
            uniform_int_distribution<> r((min + 1) / 2, max / 2);
            return r(mt) * 2;
        } else {
            // Odd
            uniform_int_distribution<> r(min / 2, (max - 1) / 2);
            return r(mt) * 2 + 1;
        }
    } else {
        uniform_int_distribution<> r(min, max);
        return r(mt);
    }
}