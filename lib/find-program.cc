#include <tuple>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <queue>
#include <sstream>
#include "dsl/utils.h"
#include "find-program.h"
#include "enumerator.h"

using namespace std;
using namespace dsl;

auto mk_calc_info() {
    return [](const Program &p, const tuple<int, bool, vector<Environment>> &info) {
        auto index = get<0>(info);
        auto env = get<2>(info);

        vector<Environment> env2;
        env2.reserve(env.size());

        for (auto i = 0; i < env.size(); i++) {
            auto e = proceed((p.at(index)), env[i]);
            if (e){
                env2.push_back(e.value());
            } else {
                return make_tuple(index + 1, false, env);
            }

        }

        return make_tuple(index + 1, true, env2);
    };
}

experimental::optional<Program> dfs(size_t max_length, const Attribute &attr, const vector<Example> &examples) {
    if (examples.size() == 0) {
        return {};
    }

    auto example = examples.at(0);
    vector<Statement> read_input;
    for (const auto &line: example.input) {
        auto var = read_input.size();
        if (line.type == Type::Integer) {
            read_input.push_back(Statement(var, Function::ReadInt, {}));
        } else {
            read_input.push_back(Statement(var, Function::ReadList, {}));
        }
    }

    Restriction r;
    r.functions = all_functions;
    r.functions.erase(find(r.functions.begin(), r.functions.end(), Function::ReadList));
    r.functions.erase(find(r.functions.begin(), r.functions.end(), Function::ReadInt));
    r.predicates = all_predicate_lambdas;
    r.one_argument_lambda = all_one_argument_lambdas;
    r.two_arguments_lambda = all_two_arguments_lambdas;

    sort(r.functions.begin(), r.functions.end(), [&](auto f1, auto f2) {
        return attr.function_presence.at(f1) > attr.function_presence.at(f2);
    });
    sort(r.predicates.begin(), r.predicates.end(), [&](auto l1, auto l2) {
        return attr.predicate_presence.at(l1) > attr.predicate_presence.at(l2);
    });
    sort(r.one_argument_lambda.begin(), r.one_argument_lambda.end(), [&](auto l1, auto l2) {
        return attr.one_argument_lambda_presence.at(l1) > attr.one_argument_lambda_presence.at(l2);
    });
    sort(r.two_arguments_lambda.begin(), r.two_arguments_lambda.end(), [&](auto l1, auto l2) {
        return attr.two_arguments_lambda_presence.at(l1) > attr.two_arguments_lambda_presence.at(l2);
    });
    r.min_length = read_input.size() + 1;
    r.max_length = read_input.size() + max_length;

    vector<Environment> initial_env;
    initial_env.reserve(examples.size());
    for (auto i = 0; i < examples.size(); i++) {
        auto example = examples.at(i);
        auto e = Environment({}, example.input);

        for (const auto &s: read_input) {
            auto x = proceed(s, e);
            if (x) {
                e = x.value();
            } else {
                return {};
            }
        }
        initial_env.push_back(e);
    }

    experimental::optional<Program> program_opt = {};
    enumerate(
            r, mk_calc_info(),
            [&](const Program &p, const tuple<int, bool, vector<Environment>> &info) {
                auto index = get<0>(info);
                auto isValid = get<1>(info);
                auto env = get<2>(info);

                if (!isValid) {
                    return true;
                }

                bool satisfy = true;
                for (auto i = 0; i < examples.size(); i++) {
                    auto expect = examples.at(i).output;
                    auto actual = env.at(i).variables.find(p.back().variable)->second;

                    if (expect != actual) {
                        satisfy = false;

                    }
                }

                if (satisfy) {
                    program_opt = p;
                }
                return !satisfy;
            },
            read_input, make_tuple(read_input.size(), true, initial_env)
    );

    return program_opt;
}

experimental::optional<Program> sort_and_add(size_t max_length, const Attribute &attr, const std::vector<Example> &examples) {
    if (examples.size() == 0) {
        return {};
    }

    auto example = examples.at(0);
    vector<Statement> read_input;
    for (const auto &line: example.input) {
        auto var = read_input.size();
        if (line.type == Type::Integer) {
            read_input.push_back(Statement(var, Function::ReadInt, {}));
        } else {
            read_input.push_back(Statement(var, Function::ReadList, {}));
        }
    }

    auto funcs = all_functions;
    funcs.erase(find(funcs.begin(), funcs.end(), Function::ReadList));
    funcs.erase(find(funcs.begin(), funcs.end(), Function::ReadInt));
    sort(funcs.begin(), funcs.end(), [&](auto f1, auto f2) {
        return attr.function_presence.at(f1) > attr.function_presence.at(f2);
    });

    auto funcs_queue = queue<Function>();
    for (auto f: funcs) {
        funcs_queue.push(f);
    }

    auto predicates = all_predicate_lambdas;
    sort(predicates.begin(), predicates.end(), [&](auto l1, auto l2) {
        return attr.predicate_presence.at(l1) > attr.predicate_presence.at(l2);
    });
    auto predicates_queue = queue<PredicateLambda>();
    for (auto p: predicates) {
        predicates_queue.push(p);
    }
    auto one_arg = all_one_argument_lambdas;
    auto two_args = all_two_arguments_lambdas;
    sort(one_arg.begin(), one_arg.end(), [&](auto l1, auto l2) {
        return attr.one_argument_lambda_presence.at(l1) > attr.one_argument_lambda_presence.at(l2);
    });
    sort(two_args.begin(), two_args.end(), [&](auto l1, auto l2) {
        return attr.two_arguments_lambda_presence.at(l1) > attr.two_arguments_lambda_presence.at(l2);
    });
    auto one_arg_queue = queue<OneArgumentLambda>();
    for (auto f: one_arg) {
        one_arg_queue.push(f);
    }
    auto two_args_queue = queue<TwoArgumentsLambda>();
    for (auto f: two_args) {
        two_args_queue.push(f);
    }

    Restriction r;
    r.min_length = read_input.size() + 1;
    r.max_length = read_input.size() + max_length;

    vector<Environment> initial_env;
    initial_env.reserve(examples.size());
    for (auto i = 0; i < examples.size(); i++) {
        auto example = examples.at(i);
        auto e = Environment({}, example.input);

        for (const auto &s: read_input) {
            auto x = proceed(s, e);
            if (x) {
                e = x.value();
            } else {
                return {};
            }
        }
        initial_env.push_back(e);
    }

    experimental::optional<Program> program_opt = {};

    cout << "function  " << stringify(funcs_queue.front()) << endl;
    r.functions.push_back(funcs_queue.front());
    funcs_queue.pop();
    while (true) {
        enumerate(
                r, mk_calc_info(),
                [&](const Program &p, const tuple<int, bool, vector<Environment>> &info) {
                    auto index = get<0>(info);
                    auto isValid = get<1>(info);
                    auto env = get<2>(info);

                    if (!isValid) {
                        return true;
                    }

                    bool satisfy = true;
                    for (auto i = 0; i < examples.size(); i++) {
                        auto expect = examples.at(i).output;
                        auto actual = env.at(i).variables.find(p.back().variable)->second;

                        if (expect != actual) {
                            satisfy = false;
                        }
                    }

                    if (satisfy) {
                        program_opt = p;
                    }
                    return !satisfy;
                },
                read_input, make_tuple(read_input.size(), true, initial_env)
        );

        if (program_opt) {
            break;
        }

        if (funcs_queue.empty() && predicates_queue.empty() && one_arg_queue.empty() && two_args_queue.empty()) {
            break;
        }

        auto f = (funcs_queue.empty()) ? 0 : attr.function_presence.at(funcs_queue.front());
        auto p = (predicates_queue.empty()) ? 0 : attr.predicate_presence.at(predicates_queue.front());
        auto o = (one_arg_queue.empty()) ? 0 : attr.one_argument_lambda_presence.at(one_arg_queue.front());
        auto t = (two_args_queue.empty()) ? 0 : attr.two_arguments_lambda_presence.at(two_args_queue.front());
        if (f >= p && f >= o && f >= t) {
            cout << "function " << " " << stringify(funcs_queue.front()) << endl;
            r.functions.push_back(funcs_queue.front());
            funcs_queue.pop();
        } else if (p >= o && p >= t) {
            cout << "predicate" << " " << stringify(predicates_queue.front()) << endl;
            r.predicates.push_back(predicates_queue.front());
            predicates_queue.pop();
        } else if (o >= t) {
            cout << "one-arg  " << " " << stringify(one_arg_queue.front()) << endl;
            r.one_argument_lambda.push_back(one_arg_queue.front());
            one_arg_queue.pop();
        } else {
            cout << "two-args " << " " << stringify(two_args_queue.front()) << endl;
            r.two_arguments_lambda.push_back(two_args_queue.front());
            two_args_queue.pop();
        }
    }

    return program_opt;
}
