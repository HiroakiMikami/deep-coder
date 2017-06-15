#include <random>
#include <algorithm>
#include "dataset-generator.h"

using namespace std;

static random_device rnd;
static mt19937 mt(rnd());

IntegerConstraint::IntegerConstraint() : min(), max(), sign(), is_even() {}

ListConstraint::ListConstraint() : min_length(), max_length(), min(), max(), sign({{}}), is_even({{}}) {}
IntegerConstraint ListConstraint::generate_integer_constraint() const {
    IntegerConstraint c;

    c.min = this->min;
    c.max = this->max;

    uniform_int_distribution<> r1(0, this->sign.size() - 1);
    auto x1 = r1(mt);
    auto it1 = this->sign.begin();
    for (auto i = 0; i < x1; i++) {
        ++it1;
    }
    c.sign = *it1;

    uniform_int_distribution<> r2(0, this->is_even.size() - 1);
    auto x2 = r2(mt);
    auto it2 = this->is_even.begin();
    for (auto i = 0; i < x1; i++) {
        ++it2;
    }
    c.is_even = *it2;

    return c;
}

std::vector<IntegerConstraint> ListConstraint::all_constraints() const {
    std::vector<IntegerConstraint> cs;
    cs.reserve(this->sign.size() * this->is_even.size());

    for (const auto& sign: this->sign) {
        for (const auto& is_even: this->is_even) {
            IntegerConstraint c;
            c.min = this->min;
            c.max = this->max;
            c.sign = sign;
            c.is_even = is_even;

            cs.push_back(c);
        }
    }

    return cs;
}

experimental::optional<int> generate_integer(const IntegerConstraint& constraint) {
    auto min = constraint.min.value_or(-256);
    auto max = constraint.max.value_or(255);

    // sign
    if (constraint.sign) {
        if (constraint.sign.value() == Sign::Positive) {
            min = std::max(min, 1);
        } else if (constraint.sign.value() == Sign::Negative){
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

std::experimental::optional<std::vector<int>> generate_list(const ListConstraint &constraint) {
    auto min_length = std::max(constraint.min_length.value_or(0), 0);
    auto max_length = std::max(constraint.max_length.value_or(10), 0);

    if (max_length < min_length) {
        return {};
    }

    uniform_int_distribution<> l(min_length, max_length);
    auto length = l(mt);

    std::vector<int> list(length);
    for (auto &elem: list) {
        bool is_initialized = false;
        auto c = constraint.generate_integer_constraint();
        auto x = generate_integer(c);
        if (x) {
            elem = x.value();
            is_initialized = true;
        } else {
            auto cs = constraint.all_constraints();
            for (const auto &c: cs) {
                auto x = generate_integer(c);
                if (x) {
                    elem = x.value();
                    is_initialized = true;
                    break;
                }
            }
        }

        if (!is_initialized) {
            return {};
        }
    }

    return list;
}