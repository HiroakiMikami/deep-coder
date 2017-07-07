#include <iostream>
#include <random>
#include <algorithm>
#include "example-generator.h"
#include "dsl/type.h"
#include "parameters.h"

using namespace std;
using namespace dsl;

static random_device rnd;
static mt19937 mt(rnd());

IntegerConstraint::IntegerConstraint() : min(), max(), sign(), is_even() {}
pair<experimental::optional<int>, experimental::optional<int>> IntegerConstraint::range() const {
    auto min = this->min;
    auto max = this->max;

    // sign
    if (this->sign) {
        if (this->sign.value() == Sign::Positive) {
            min = std::max(min.value_or(1), 1);
        } else if (this->sign.value() == Sign::Negative){
            max = std::min(max.value_or(-1), -1);
        } else {
            return {0, 0};
        }
    }
    return {min, max};
};

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
    for (auto i = 0; i < x2; i++) {
        ++it2;
    }
    c.is_even = *it2;

    return c;
}

vector<IntegerConstraint> ListConstraint::all_constraints() const {
    vector<IntegerConstraint> cs;
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
    auto p = constraint.range();
    auto min = p.first.value_or(INPUT_MIN);
    auto max = p.second.value_or(INPUT_MAX);

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

experimental::optional<vector<int>> generate_list(const ListConstraint &constraint) {
    auto min_length = std::max(constraint.min_length.value_or(0), 0);
    auto max_length = std::max(constraint.max_length.value_or(LIST_LENGTH), 0);

    if (max_length < min_length) {
        return {};
    }

    uniform_int_distribution<> l(min_length, max_length);
    auto length = l(mt);

    vector<int> list(length);
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

experimental::optional<Constraint> analyze(const Program &p) {
    auto tenv_ = generate_type_environment(p);
    if (!tenv_) {
        return {};
    }

    auto tenv = tenv_.value();
    Constraint c;

    auto list_constraint = [&c](const Argument& arg) -> ListConstraint& {
        auto var = arg.variable().value();
        if (c.list_variables.find(var) == c.list_variables.end()) {
            c.list_variables.insert({var, ListConstraint()});
        }
        return c.list_variables.find(var)->second;
    };
    auto integer_constraint = [&](const Argument &arg) -> IntegerConstraint& {
        auto var = arg.variable().value();
        if (c.integer_variables.find(var) == c.integer_variables.end()) {
            c.integer_variables.insert({var, IntegerConstraint()});
        }
        return c.integer_variables.find(var)->second;
    };

    for (auto it = p.rbegin(); it != p.rend(); it++) {
        // Generate constraints of arguments from it->variable's constraint
        auto var = it->variable;
        auto t = tenv.find(var)->second;

        if (t == Type::Integer) {
            auto ic = integer_constraint(Argument(var));

            if (it->function == Function::Head || it->function == Function::Last) {
                auto &lc = list_constraint(it->arguments.at(0));
                lc.sign.insert(ic.sign);
                lc.is_even.insert(ic.is_even);
                lc.min_length = max(lc.min_length.value_or(0), 1);
            } else if (it->function == Function::Access) {
                auto &nc = integer_constraint(it->arguments.at(0));
                auto &lc = list_constraint(it->arguments.at(1));
                nc.min = max(nc.min.value_or(0), 0);
                lc.min_length = max(lc.min_length.value_or(0), nc.min.value_or(0));
                lc.sign.insert(ic.sign);
                lc.is_even.insert(ic.is_even);
                lc.min_length = max(lc.min_length.value_or(0), 1);
            } else if (it->function == Function::Maximum) {
                auto &lc = list_constraint(it->arguments.at(0));
                lc.min_length = max(lc.min_length.value_or(1), 1);
                if (ic.max) {
                    lc.max = min(lc.max.value_or(ic.max.value()), ic.max.value());
                }
            } else if (it->function == Function::Minimum) {
                auto &lc = list_constraint(it->arguments.at(0));
                lc.min_length = max(lc.min_length.value_or(1), 1);
                if (ic.min) {
                    lc.min = max(lc.min.value_or(ic.min.value()), ic.min.value());
                }
            } else if (it->function == Function::Sum) {
                auto &lc = list_constraint(it->arguments.at(0));
                auto range = ic.range();
                if (range.first.value_or(0) > 0 || range.second.value_or(0) < 0) {
                    lc.min_length = max(lc.min_length.value_or(0), 1);
                }
            } else if (it->function == Function::Count) {
                auto lambda = it->arguments.at(0).predicate().value();
                auto &lc = list_constraint(it->arguments.at(1));

                if (ic.min) {
                    lc.min_length = max(lc.min_length.value_or(0), ic.min.value());

                    if (ic.min.value() >= 1) {
                        switch (lambda) {
                            case PredicateLambda::IsPositive:
                                lc.sign.insert(Sign::Positive);
                                break;
                            case PredicateLambda::IsNegative:
                                lc.sign.insert(Sign::Negative);
                                break;
                            case PredicateLambda::IsOdd:
                                lc.is_even.insert(false);
                                break;
                            case PredicateLambda::IsEven:
                                lc.is_even.insert(true);
                                break;
                        }
                    }
                }
            } else if (it->function == Function::ReadInt) {
                c.inputs.push_back(var);
            } else {
                cerr << "Implementation Error" << endl;
            }
        } else {
            auto &lc = list_constraint(Argument(var));

            if (it->function == Function::Take || it->function == Function::Drop) {
                auto &nc = integer_constraint(it->arguments.at(0));
                auto &lc2 = list_constraint(it->arguments.at(1));

                nc.min = max(nc.min.value_or(0), 0);
                for (const auto& x: lc.sign) {
                    lc2.sign.insert(x);
                }
                for (const auto& x: lc.is_even) {
                    lc2.is_even.insert(x);
                }
                if (lc.min_length) {
                    lc2.min_length = max(lc.min_length.value(), lc2.min_length.value_or(0));
                }
            } else if (it->function == Function::Reverse || it->function == Function::Sort) {
                auto &lc2 = list_constraint(it->arguments.at(0));
                lc2 = lc;
            } else if (it->function == Function::Map) {
                auto lambda = it->arguments.at(0).one_argument_lambda().value();
                auto &lc2 = list_constraint(it->arguments.at(1));

                if (lc.min_length) {
                    lc2.min_length = max(lc2.min_length.value_or(0), lc.min_length.value());
                }

                switch (lambda) {
                    case OneArgumentLambda::Plus1:
                        if (lc.min) {
                            lc2.min = lc.min.value() - 1;
                        }
                        if (lc.max) {
                            lc2.max = lc.max.value() - 1;
                        }
                        for (const auto &x: lc.is_even) {
                            lc2.is_even.insert(!x);
                        }
                        break;
                    case OneArgumentLambda::Minus1:
                        if (lc.min) {
                            lc2.min = lc.min.value() + 1;
                        }
                        if (lc.max) {
                            lc2.max = lc.max.value() + 1;
                        }

                        lc2.is_even.clear();
                        for (const auto &x: lc.is_even) {
                            lc2.is_even.insert(!x);
                        }
                        break;
                    case OneArgumentLambda::MultiplyMinus1:
                        if (lc.max) {
                            lc2.min = lc.max.value() * (-1);
                        }
                        if (lc.min) {
                            lc2.max = lc.min.value() * (-1);
                        }
                        lc2.is_even = lc.is_even;
                        for (const auto &x: lc.sign) {
                            if (x == Sign::Positive) {
                                lc2.sign.insert(Sign::Negative);
                            } else if (x == Sign::Negative) {
                                lc2.sign.insert(Sign::Positive);
                            } else {
                                lc2.sign.insert(x);
                            }
                        }
                        break;
                    case OneArgumentLambda::Multiply2:
                        if (lc.min) {
                            lc2.min = lc.min.value() / 2;
                        }
                        if (lc.min) {
                            lc2.max = lc.max.value() / 2;
                        }
                        lc2.sign = lc.sign;
                        break;
                    case OneArgumentLambda::Multiply3:
                        if (lc.min) {
                            lc2.min = lc.min.value() / 3;
                        }
                        if (lc.min) {
                            lc2.max = lc.max.value() / 3;
                        }
                        lc2.sign = lc.sign;
                        lc2.is_even = lc.is_even;
                        break;
                    case OneArgumentLambda::Multiply4:
                        if (lc.min) {
                            lc2.min = lc.min.value() / 4;
                        }
                        if (lc.min) {
                            lc2.max = lc.max.value() / 4;
                        }
                        lc2.sign = lc.sign;
                        break;
                    case OneArgumentLambda::Divide2:
                        if (lc.min) {
                            lc2.min = lc.min.value() / 2;
                        }
                        if (lc.min) {
                            lc2.max = lc.max.value() / 2;
                        }
                        lc2.sign = lc.sign;
                        break;
                    case OneArgumentLambda::Divide3:
                        if (lc.min) {
                            lc2.min = lc.min.value() / 3;
                        }
                        if (lc.min) {
                            lc2.max = lc.max.value() / 3;
                        }
                        lc2.sign = lc.sign;
                        lc2.is_even = lc.is_even;
                        break;
                    case OneArgumentLambda::Divide4:
                        if (lc.min) {
                            lc2.min = lc.min.value() / 4;
                        }
                        if (lc.min) {
                            lc2.max = lc.max.value() / 4;
                        }
                        lc2.sign = lc.sign;
                        break;
                    case OneArgumentLambda::Pow2:
                        // TODO
                        break;
                    default:
                        break;
                }
            } else if (it->function == Function::Filter) {
                auto lambda = it->arguments.at(0).predicate().value();
                auto &lc2 = list_constraint(it->arguments.at(1));

                if (lc.min_length) {
                    lc2.min_length = max(lc2.min_length.value_or(0), lc.min_length.value());

                    if (lc.min_length.value() >= 1) {
                        switch (lambda) {
                            case PredicateLambda::IsPositive:
                                lc2.sign.insert(Sign::Positive);
                                break;
                            case PredicateLambda::IsNegative:
                                lc2.sign.insert(Sign::Negative);
                                break;
                            case PredicateLambda::IsOdd:
                                lc2.is_even.insert(false);
                                break;
                            case PredicateLambda::IsEven:
                                lc2.is_even.insert(true);
                                break;
                        }
                    }
                }
            } else if (it->function == Function::ZipWith) {
                auto &lc2 = list_constraint(it->arguments.at(1));
                auto &lc3 = list_constraint(it->arguments.at(2));

                if (lc.min_length) {
                    lc2.min_length = max(lc2.min_length.value_or(0), lc.min_length.value());
                    lc3.min_length = max(lc3.min_length.value_or(0), lc.min_length.value());
                }
            } else if (it->function == Function::Scanl1) {
                auto &lc2 = list_constraint(it->arguments.at(1));

                if (lc.min_length) {
                    lc2.min_length = max(lc2.min_length.value_or(0), lc.min_length.value());
                }
            } else if (it->function == Function::ReadList) {
                c.inputs.push_back(it->variable);
            } else {
                cerr << "Implementation Error" << endl;
            }
        }
    }

    reverse(c.inputs.begin(), c.inputs.end());

    return c;
}

bool is_in_range(const Value &v) {
    if (v.integer()) {
        auto i = v.integer().value();
        return i >= INTEGER_MIN && i <= INTEGER_MAX;
    } else if (v.list()) {
        auto l = v.list().value();
        return all_of(l.begin(), l.end(), [](const auto &i) {
            return i >= INTEGER_MIN && i <= INTEGER_MAX;
        });
    } else {
        return true;
    }
}

experimental::optional<vector<Example>> generate_examples(const dsl::Program &p, size_t example_num) {
    auto c_ = analyze(p);
    if (!c_) {
        return {};
    }

    auto c = c_.value();

    vector<Example> examples;
    examples.reserve(example_num);

    for (auto i = 0; i < example_num * 100; i++) {
        // Generate inputs
        Input input;

        for (const auto& input_var: c.inputs) {
            if (c.integer_variables.find(input_var) != c.integer_variables.end()) {
                auto constraint = c.integer_variables.find(input_var)->second;
                auto n = generate_integer(constraint);
                if (n) {
                    input.push_back(Value(n.value()));
                } else {
                    break;
                }
            } else {
                auto constraint = c.list_variables.find(input_var)->second;
                auto l = generate_list(constraint);
                if (l) {
                    input.push_back(Value(l.value()));
                } else {
                    break;
                }
            }
        }

        if (c.inputs.size() == input.size()) {
            auto output = eval(p, input);
            if (output && !output.value().is_null()) {
                auto o = output.value();

                if (!is_in_range(o)) {
                    continue ;
                }

                examples.push_back(Example{input, output.value()});

                if (examples.size() >= example_num) {
                    break;
                }
            }
        }
    }

    return examples;
}