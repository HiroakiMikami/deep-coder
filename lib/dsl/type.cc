#include "dsl/type.h"
#include <iostream>

using namespace std;

namespace dsl {
    experimental::optional<Type> get_type(const Argument& argument, const TypeEnvironment& env) {
        if (argument.one_argument_lambda()) {
            return Type::OneArgumentLambda;
        } else if (argument.two_arguments_lambda()) {
            return Type::TwoArgumentsLambda;
        } else if (argument.predicate()) {
            return Type::PredicateLambda;
        } else if (argument.variable()) {
            auto it = env.find(argument.variable().value());
            if (it != env.end()) {
                return it->second;
            }
        }
        return {};
    }

    Signature get_signature(Function function) {
        switch (function) {
            case Function::Head:
                return {Type::Integer, {Type::List}};
            case Function::Last:
                return {Type::Integer, {Type::List}};
            case Function::Take:
                return {Type::List, {Type::Integer, Type::List}};
            case Function::Drop:
                return {Type::List, {Type::Integer, Type::List}};
            case Function::Access:
                return {Type::Integer, {Type::Integer, Type::List}};
            case Function::Minimum:
                return {Type::Integer, {Type::List}};
            case Function::Maximum:
                return {Type::Integer, {Type::List}};
            case Function::Reverse:
                return {Type::List, {Type::List}};
            case Function::Sort:
                return {Type::List, {Type::List}};
            case Function::Sum:
                return {Type::Integer, {Type::List}};
            case Function::Map:
                return {Type::List, {Type::OneArgumentLambda, Type::List}};
            case Function::Filter:
                return {Type::List, {Type::PredicateLambda, Type::List}};
            case Function::Count:
                return {Type::Integer, {Type::PredicateLambda, Type::List}};
            case Function::ZipWith:
                return {Type::List, {Type::TwoArgumentsLambda, Type::List, Type::List}};
            case Function::Scanl1:
                return {Type::List, {Type::TwoArgumentsLambda, Type::List}};
            case Function::ReadInt:
                return {Type::Integer, {}};
            case Function::ReadList:
                return {Type::List, {}};
        }
    }

    experimental::optional<TypeEnvironment> check(const Statement &statement, const TypeEnvironment &env) {
        auto signature = get_signature(statement.function);
        // Check arguments type
        if (signature.arguments_type.size() != statement.arguments.size()) {
            return {};
        }

        for (auto i = 0; i < signature.arguments_type.size(); i++) {
            auto expected_type = signature.arguments_type[i];
            auto argument = statement.arguments[i];

            auto actual_type = get_type(argument, env);

            if (expected_type != actual_type) {
                return {};
            }
        }

        // Add variable type
        if (env.find(statement.variable) != env.end()) {
            return {};
        }

        auto next = env;
        next.insert({statement.variable, signature.return_type});
        return next;
    }

    std::experimental::optional<TypeEnvironment> generate_type_environment(const Program &program) {
        TypeEnvironment env;
        for (const auto &statement: program) {
            auto next_env = check(statement, env);
            if (!next_env) {
                return {};
            }
            env = next_env.value();
        }
        return env;
    }

    bool is_valid(const Program &program) {
        return static_cast<bool>(generate_type_environment(program));
    }
}