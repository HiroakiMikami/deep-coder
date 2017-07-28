#include <vector>
#include <stack>
#include <random>
#include <cassert>
#include "enumerator.h"
#include "dsl/ast.h"
#include "dsl/type.h"

#pragma once

std::experimental::optional<dsl::Program> generate_random_program(const Restriction &restriction, const dsl::Program &p, const dsl::TypeEnvironment t) {
    dsl::Program program = p;
    dsl::TypeEnvironment tenv = t;

    static std::random_device rnd;
    static std::mt19937 mt(rnd());
    std::uniform_int_distribution<> func(0, restriction.functions.size() - 1);
    std::uniform_int_distribution<> len(restriction.min_length, restriction.max_length);


    auto length = len(mt);

    for (auto i = 0; i < length; i++) {
        auto f = restriction.functions[func(mt)];
        auto arg_types = dsl::get_signature(f).arguments_type;
        auto as = std::vector<std::vector<dsl::Argument>>(arg_types.size());
        for (auto i = 0; i < arg_types.size(); i++) {
            auto t = arg_types[i];
            switch (t) {
                case dsl::Type::PredicateLambda:
                    as[i].reserve(restriction.predicates.size());
                    for (const auto &p: restriction.predicates) {
                        as[i].push_back(p);
                    }
                    break;
                case dsl::Type::OneArgumentLambda:
                    as[i].reserve(restriction.one_argument_lambda.size());
                    for (const auto &l: restriction.one_argument_lambda) {
                        as[i].push_back(l);
                    }
                    break;
                case dsl::Type::TwoArgumentsLambda:
                    as[i].reserve(restriction.two_arguments_lambda.size());
                    for (const auto &l: restriction.two_arguments_lambda) {
                        as[i].push_back(l);
                    }
                    break;
                case dsl::Type::Integer:
                    for (const auto &var: tenv) {
                        if (var.second == dsl::Type::Integer) {
                            as[i].push_back(dsl::Argument(var.first));
                        }
                    }
                    break;
                case dsl::Type::List:
                    for (const auto &var: tenv) {
                        if (var.second == dsl::Type::List) {
                            as[i].push_back(dsl::Argument(var.first));
                        }
                    }
                    break;
                default:
                    break;
            }
        }

        auto s = std::stack<std::vector<dsl::Argument>>();
        if (as.size() == 0) {
            auto s = dsl::Statement(program.size(), f, {});
            program.push_back(s);
            tenv = dsl::check(s, tenv).value();
        } else {
            std::vector<dsl::Argument> args;
            for (auto i = 0; i < as.size(); i++) {
                if (as[i].size() == 0) {
                    return {};
                }
                std::uniform_int_distribution<> a(0, as[i].size() - 1);
                auto x = a(mt);
                args.push_back(as[i][x]);
            }
            auto s = dsl::Statement(program.size(), f, args);
            program.push_back(s);
            tenv = dsl::check(s, tenv).value();
        }
    }

    return program;
}

std::experimental::optional<dsl::Program> generate_random_program(const Restriction &restriction) {
    return generate_random_program(restriction, {}, {});
}
