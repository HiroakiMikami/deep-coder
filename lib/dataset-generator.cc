#include <iostream>
#include <algorithm>
#include <unordered_set>
#include "dataset-generator.h"
#include "example-generator.h"
#include "enumerator.h"

using namespace std;
using namespace dsl;

bool has_unused_variable(const dsl::Program &p) {
    unordered_set<Variable> unused_var;
    unused_var.reserve(p.size());

    for (const auto &s: p) {
        for (const auto &arg: s.arguments) {
            if (arg.variable()) {
                auto var = arg.variable().value();
                unused_var.erase(var);
            }
        }

        unused_var.insert(s.variable);
    }

    return unused_var.size() > 1;
}

Dataset::Dataset() : size(0) {}
void Dataset::insert(const Program &p, const vector<Example> &examples) {
    if (examples.size() < 1) {
        return ;
    }

    // Check type of program
    ProgramType type;
    auto example = examples[0];
    type.first.reserve(example.input.size());
    for (const auto &i: example.input) {
        if (i.integer()) {
            type.first.push_back(Type::Integer);
        } else {
            type.first.push_back(Type::List);
        }
    }

    if (example.output.integer()) {
        type.second = Type::Integer;
    } else {
        type.second = Type::List;
    }

    // Search equivalent program
    if (this->programs.find(type) == this->programs.end()) {
        this->programs.insert({type, {}});
    }

    auto candidates = this->programs.find(type);
    bool has_equivalent_program = false;
    vector<int> indexes_to_be_deleted;
    indexes_to_be_deleted.reserve(candidates->second.size());
    size_t deleted_size = 0;
    for (auto i = 0; i < candidates->second.size(); i++) {
        const auto &candidate = candidates->second.at(i);
        bool is_equivalent = true;
        for (const auto &example: examples) {
            auto output = eval(candidate.first, example.input);
            if (!output) {
                is_equivalent = false;
            } else {
                if (output.value() != example.output) {
                    is_equivalent = false;
                }
            }
        }

        if (is_equivalent) {
            if (candidate.first.size() > p.size()) {
                indexes_to_be_deleted.push_back(i);
                deleted_size += candidate.second.size();
            } else {
                has_equivalent_program = true;
            }
        }
    }

    if (!has_equivalent_program) {
        this->programs.find(type)->second.push_back({p, examples});
        this->size += examples.size();
    }

    for_each(indexes_to_be_deleted.rbegin(), indexes_to_be_deleted.rend(), [&candidates](const auto &i) {
        candidates->second.erase(candidates->second.begin() + i);
    });
    this->size -= deleted_size;
}

experimental::optional<Dataset> generate_dataset(size_t min_length, size_t max_length, size_t dataset_size) {
    auto functions = all_functions;
    functions.erase(find(functions.begin(), functions.end(), Function::ReadInt));
    functions.erase(find(functions.begin(), functions.end(), Function::ReadList));

    // Enumerate read_{list, int}
    Restriction r_for_read;
    r_for_read.min_length = 1;
    r_for_read.max_length = max_length - 1;
    r_for_read.functions = { Function::ReadInt, Function::ReadList };

    Restriction r;
    r.min_length = 1;
    r.max_length = max_length;
    r.functions = all_functions;

    auto calc_info = [](const Program& p, const int &i) { return i; };

    Dataset dataset;

    enumerate(
            r_for_read, calc_info,
            [&r, &calc_info, &dataset, &dataset_size](const Program &p, const int &i) -> bool {
                enumerate(
                        r, calc_info,
                        [&dataset, &dataset_size](const Program &p, const int &i) -> bool {
                            // Check program
                            //// Unused program
                            if (has_unused_variable(p)) {
                                return true;
                            }

                            // Generate example
                            auto examples_ = generate_examples(p);

                            if (!examples_) {
                                return true;
                            }

                            auto examples = examples_.value();

                            dataset.insert(p, examples);

                            cerr << "Generating dataset... " << dataset.size;
                            if (dataset_size != 0) {
                                cout << " / " << dataset_size;
                            }
                            cout << endl;

                            if (dataset_size == 0) {
                                return true;
                            } else {
                                return dataset.size < dataset_size;
                            }
                        },
                        i
                );

                if (dataset_size == 0) {
                    return true;
                } else {
                    return dataset.size < dataset_size;
                }
            },
            0
    );

    return dataset;
}