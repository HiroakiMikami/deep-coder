#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include "dsl/type.h"
#include "dsl/interpreter.h"

using namespace std;

namespace dsl {
    function<int(const int&)> get_one_argument_lambda(OneArgumentLambda lambda) {
        switch (lambda) {
            case OneArgumentLambda::Plus1: return [](const int& x) { return x + 1; };
            case OneArgumentLambda::Minus1: return [](const int& x) { return x - 1; };
            case OneArgumentLambda::MultiplyMinus1: return [](const int& x) { return x * (-1); };
            case OneArgumentLambda::Multiply2: return [](const int& x) { return x * 2; };
            case OneArgumentLambda::Multiply3: return [](const int& x) { return x * 3; };
            case OneArgumentLambda::Multiply4: return [](const int& x) { return x * 4; };
            case OneArgumentLambda::Divide2: return [](const int& x) { return x / 2; };
            case OneArgumentLambda::Divide3: return [](const int& x) { return x / 3; };
            case OneArgumentLambda::Divide4: return [](const int& x) { return x / 4; };
            case OneArgumentLambda::Pow2: return [](const int& x) { return x * x; };
        }
    }
    function<int(const int&, const int&)> get_two_arguments_lambda(TwoArgumentsLambda lambda) {
        switch(lambda) {
            case TwoArgumentsLambda::Plus: return std::plus<int>();
            case TwoArgumentsLambda::Minus: return std::minus<int>();
            case TwoArgumentsLambda::Multiply: return std::multiplies<int>();
            case TwoArgumentsLambda::Min: return [](const int& x, const int& y) { return min(x, y); };
            case TwoArgumentsLambda::Max: return [](const int& x, const int& y) { return max(x, y); };
        }
    }
    function<bool(const int&)> get_predicate_lambda(PredicateLambda lambda) {
        switch (lambda) {
            case PredicateLambda::IsPositive: return [](const int& x) { return x > 0; };
            case PredicateLambda::IsNegative: return [](const int& x) { return x < 0; };
            case PredicateLambda::IsEven: return [](const int& x) { return (x % 2) == 0; };
            case PredicateLambda::IsOdd: return [](const int& x) { return std::abs(x % 2) == 1; };
        }
    }

    Value::Value() : type(Type::Null), result(0) {}

    Value::Value(int i) : type(Type::Integer), result({i}) {}

    Value::Value(const std::vector<int> &l) : type(Type::List), result(l) {}

    std::experimental::optional<int> Value::integer() const {
        if (this->type == Type::Integer && this->result.size() >= 1) {
            return this->result[0];
        } else {
            return {};
        }
    }

    std::experimental::optional<std::vector<int>> Value::list() const {
        if (this->type == Type::List) {
            return this->result;
        } else {
            return {};
        }
    }

    bool Value::is_null() const {
        return this->type == Type::Null;
    }

    bool Value::operator==(const Value& rhs) const {
        return (this->type == rhs.type) && (this->result == rhs.result);
    }
    bool Value::operator!=(const Value& rhs) const {
        return !(*this == rhs);
    }


    Environment::Environment(const std::unordered_map<Variable, Value> &variables, const Input& input)
            : variables(variables), input(input), offset(0) {}

    Value eval(Function function, const vector<Argument> &arguments, Environment &environment) {
        if (function == Function::Head) {
            auto arg = environment.variables.find(arguments[0].variable().value())->second;

            if (arg.list()) {
                auto l = arg.list().value();
                return (l.size() == 0) ? Value() : Value(l[0]);
            } else {
                return Value();
            }
        } else if (function == Function::Last) {
            auto arg = environment.variables.find(arguments[0].variable().value())->second;

            if (arg.list()) {
                auto l = arg.list().value();
                return (l.size() == 0) ? Value() : Value(l[l.size() - 1]);
            } else {
                return Value();
            }
        } else if (function == Function::Take) {
            auto n = environment.variables.find(arguments[0].variable().value())->second;
            auto list = environment.variables.find(arguments[1].variable().value())->second;

            if (n.integer() && list.list()) {
                auto l = list.list().value();

                if (n.integer().value() < 0) {
                    return Value(vector<int>());
                }

                auto num = min(static_cast<size_t>(n.integer().value()), l.size());
                vector<int> retval;
                retval.reserve(num);
                for (auto i = 0; i < num; i++) {
                    retval.push_back(l[i]);
                }
                return Value(retval);
            } else {
                return Value();
            }
        } else if (function == Function::Drop) {
            auto n = environment.variables.find(arguments[0].variable().value())->second;
            auto list = environment.variables.find(arguments[1].variable().value())->second;

            if (n.integer() && list.list()) {
                auto l = list.list().value();

                if (n.integer().value() < 0) {
                    return Value(vector<int>());
                }

                auto num = static_cast<size_t>(n.integer().value());
                vector<int> retval;
                if (l.size() > num) {
                    retval.reserve(l.size() - num);
                }
                for (auto i = num; i < l.size(); i++) {
                    retval.push_back(l[i]);
                }
                return Value(retval);
            } else {
                return Value();
            }
        } else if (function == Function::Access) {
            auto n = environment.variables.find(arguments[0].variable().value())->second;
            auto list = environment.variables.find(arguments[1].variable().value())->second;

            if (n.integer() && list.list()) {
                auto l = list.list().value();
                auto i = static_cast<size_t>(n.integer().value());

                return (0 <= i && i < l.size()) ? Value(l[i]) : Value();
            } else {
                return Value();
            }
        } else if (function == Function::Minimum) {
            auto arg = environment.variables.find(arguments[0].variable().value())->second;

            if (arg.list()) {
                auto l = arg.list().value();
                if (l.size() == 0) {
                    return Value();
                }

                return *(min_element(l.begin(), l.end()));
            } else {
                return Value();
            }
        } else if (function == Function::Maximum) {
            auto arg = environment.variables.find(arguments[0].variable().value())->second;

            if (arg.list()) {
                auto l = arg.list().value();
                if (l.size() == 0) {
                    return Value();
                }

                return *(max_element(l.begin(), l.end()));
            } else {
                return Value();
            }
        } else if (function == Function::Reverse) {
            auto arg = environment.variables.find(arguments[0].variable().value())->second;

            if (arg.list()) {
                auto l = arg.list().value();
                reverse(l.begin(), l.end());
                return l;
            } else {
                return Value();
            }
        } else if (function == Function::Sort) {
            auto arg = environment.variables.find(arguments[0].variable().value())->second;

            if (arg.list()) {
                auto l = arg.list().value();
                sort(l.begin(), l.end());
                return l;
            } else {
                return Value();
            }
        } else if (function == Function::Sum) {
            auto arg = environment.variables.find(arguments[0].variable().value())->second;

            if (arg.list()) {
                auto l = arg.list().value();

                return accumulate(l.begin(), l.end(), 0);
            } else {
                return Value();
            }
        } else if (function == Function::Map) {
            auto lambda = arguments[0].one_argument_lambda().value();
            auto list = environment.variables.find(arguments[1].variable().value())->second;

            if (list.list()) {
                auto l = list.list().value();
                auto f = get_one_argument_lambda(lambda);

                for (auto& x: l) {
                    x = f(x);
                }
                return l;
            } else {
                return Value();
            }
        } else if (function == Function::Filter) {
            auto lambda = arguments[0].predicate().value();
            auto list = environment.variables.find(arguments[1].variable().value())->second;

            if (list.list()) {
                auto l = list.list().value();
                auto f = get_predicate_lambda(lambda);

                auto tmp = vector<int>();
                tmp.reserve(l.size());

                for (const auto& x: l) {
                    if (f(x)) {
                        tmp.push_back(x);
                    }
                }
                return tmp;
            } else {
                return Value();
            }
        } else if (function == Function::Count) {
            auto lambda = arguments[0].predicate().value();
            auto list = environment.variables.find(arguments[1].variable().value())->second;

            if (list.list()) {
                auto l = list.list().value();
                auto f = get_predicate_lambda(lambda);

                int n = 0;
                for (const auto& x: l) {
                    if (f(x)) {
                        n++;
                    }
                }
                return n;
            } else {
                return Value();
            }
        } else if (function == Function::ZipWith) {
            auto lambda = arguments[0].two_arguments_lambda().value();
            auto list1 = environment.variables.find(arguments[1].variable().value())->second;
            auto list2 = environment.variables.find(arguments[2].variable().value())->second;

            if (list1.list() && list2.list()) {
                auto l1 = list1.list().value();
                auto l2 = list2.list().value();
                auto f = get_two_arguments_lambda(lambda);

                vector<int> tmp(min(l1.size(), l2.size()));

                for (auto i = 0; i < tmp.size(); i++) {
                    tmp[i] = f(l1[i], l2[i]);
                }
                return tmp;
            } else {
                return Value();
            }
        } else if (function == Function::Scanl1) {
            auto lambda = arguments[0].two_arguments_lambda().value();
            auto list = environment.variables.find(arguments[1].variable().value())->second;

            if (list.list()) {
                auto l = list.list().value();
                auto f = get_two_arguments_lambda(lambda);

                if (l.size() == 0) {
                    return vector<int>(0);
                }

                vector<int> tmp(l.size());

                for (auto i = 0; i < tmp.size(); i++) {
                    if (i == 0) {
                        tmp[i] = l[i];
                    } else {
                        tmp[i] = f(tmp[i - 1], l[i]);
                    }
                }
                return tmp;
            } else {
                return Value();
            }
        } else if (function == Function::ReadInt) {
            auto value = environment.input[environment.offset];
            environment.offset += 1;

            return value.integer() ? value : Value();
        } else if (function == Function::ReadList) {
            auto value = environment.input[environment.offset];
            environment.offset += 1;

            return value.list() ? value : Value();
        }
        return Value();
    }

    experimental::optional <Environment> proceed(const Statement &statement, const Environment &environment) {
        auto next = environment;

        auto value = eval(statement.function, statement.arguments, next);
        next.variables.insert({statement.variable, value});

        return next;
    }

    experimental::optional <Output> eval(const Program &program, const Input &input) {
        auto env = Environment({}, input);
        for (const auto& s: program) {
            auto next = proceed(s, env);
            if (!next) {
                return {};
            }
            env = next.value();
        }

        if (program.size() == 0) {
            return {};
        }
        return env.variables.find(program.back().variable)->second;
    }
}