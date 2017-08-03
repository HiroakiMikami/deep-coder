#include "dsl/ast.h"
#include "attribute.h"
#include "dsl/utils.h"
#include <iostream>
using namespace std;

using namespace dsl;

size_t to_size_t(const Function &f) {
    return static_cast<size_t>(f);
}
size_t to_size_t(const PredicateLambda &l) {
    return static_cast<size_t>(l) - static_cast<size_t>(dsl::PredicateLambda::IsPositive);
}
size_t to_size_t(const OneArgumentLambda &l) {
    return static_cast<size_t>(l) - static_cast<size_t>(dsl::OneArgumentLambda::Plus1);
}
size_t to_size_t(const TwoArgumentsLambda &l) {
    return static_cast<size_t>(l) - static_cast<size_t>(dsl::TwoArgumentsLambda::Plus);
}

Attribute::Attribute(const dsl::Program &program) {
    for (const auto& f: all_functions) {
        if (f == Function::ReadInt || f == Function::ReadList) {
            continue ;
        }

        this->function_presence.insert({f, 0});
    }
    for (const auto &l: all_predicate_lambdas) {
        this->predicate_presence.insert({l, 0});
    }
    for (const auto &l: all_one_argument_lambdas) {
        this->one_argument_lambda_presence.insert({l, 0});
    }
    for (const auto &l: all_two_arguments_lambdas) {
        this->two_arguments_lambda_presence.insert({l, 0});
    }

    for (const auto& s: program) {
        if (s.function == Function::ReadInt || s.function == Function::ReadList) {
            continue ;
        }

        this->function_presence[s.function] = 1;
        for (const auto& arg: s.arguments) {
            if (arg.predicate()) {
                this->predicate_presence[arg.predicate().value()] = 1;
            } else if (arg.one_argument_lambda()) {
                this->one_argument_lambda_presence[arg.one_argument_lambda().value()] = 1;
            } else if (arg.two_arguments_lambda()) {
                this->two_arguments_lambda_presence[arg.two_arguments_lambda().value()] = 1;
            }
        }
    }
}
Attribute::Attribute(const std::vector<double> &attribute_vector) {
    auto total = 0;
    for (auto i = 0; i < all_functions.size() - 2; i++) {
        this->function_presence.insert({static_cast<Function>(i), attribute_vector[i]});
    }
    total += this->function_presence.size();

    for (auto i = 0; i < all_predicate_lambdas.size(); i++) {
        auto x = i + static_cast<size_t>(PredicateLambda::IsPositive);
        this->predicate_presence.insert({static_cast<PredicateLambda>(x), attribute_vector[i + total]});
    }
    total += this->predicate_presence.size();
    for (auto i = 0; i < all_one_argument_lambdas.size(); i++) {
        auto x = i + static_cast<size_t>(OneArgumentLambda::Plus1);
        this->one_argument_lambda_presence.insert({static_cast<OneArgumentLambda>(x), attribute_vector[i + total]});
    }
    total += this->one_argument_lambda_presence.size();
    for (auto i = 0; i < all_two_arguments_lambdas.size(); i++) {
        auto x = i + static_cast<size_t>(TwoArgumentsLambda::Plus);
        this->two_arguments_lambda_presence.insert({static_cast<TwoArgumentsLambda>(x), attribute_vector[i + total]});
    }
}

Attribute::operator std::vector<double>() const {
    std::vector<double> retval(
            all_functions.size() - 2 + all_predicate_lambdas.size() +
                    all_one_argument_lambdas.size() + all_two_arguments_lambdas.size()
    );

    auto total = 0;
    for (const auto &elem: this->function_presence) {
        retval[to_size_t(elem.first)] = elem.second;
    }
    total += this->function_presence.size() - 2;

    for (const auto &elem: this->predicate_presence) {
        retval[to_size_t(elem.first) + total] = elem.second;
    }
    total += this->predicate_presence.size();

    for (const auto &elem: this->one_argument_lambda_presence) {
        retval[to_size_t(elem.first) + total] = elem.second;
    }
    total += one_argument_lambda_presence.size();

    for (const auto &elem: this->two_arguments_lambda_presence) {
        retval[to_size_t(elem.first) + total] = elem.second;
    }

    return retval;
}
