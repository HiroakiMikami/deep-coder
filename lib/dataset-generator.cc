#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <future>
#include "dsl/utils.h"
#include "dataset-generator.h"
#include "enumerator.h"
#include "random-enumerator.h"

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


DatasetForOneInputType::DatasetForOneInputType() : size(0) {
    void insert(const dsl::Program &p, const std::vector<Example> &examples);
};
void DatasetForOneInputType::insert(const Program &p, const vector<Example> &examples) {
    if (examples.size() < EXAMPLE_NUM) {
        return ;
    }

    // Check type of the program output
    auto example = examples[0];
    auto &candidates = (example.output.integer())
                      ? this->int_output_programs
                      : this->list_output_programs;

    // Search equivalent program
    bool has_equivalent_program = false;
    vector<int> indexes_to_be_deleted;
    indexes_to_be_deleted.reserve(candidates.size());
    size_t deleted_size = 0;
    for (auto i = 0; i < candidates.size(); i++) {
        const auto &candidate = candidates.at(i);
        bool is_equivalent = true;
        for (const auto &example: candidate.second) {
            auto output = eval(p, example.input);
            if (!output) {
                is_equivalent = false;
                break;
            } else {
                if (output.value() != example.output) {
                    is_equivalent = false;
                    break;
                }
            }
        }
        if (is_equivalent) {
            for (const auto &example: examples) {
                auto output = eval(candidate.first, example.input);
                if (!output) {
                    is_equivalent = false;
                    break ;
                } else {
                    if (output.value() != example.output) {
                        is_equivalent = false;
                        break ;
                    }
                }
            }
        }

        if (is_equivalent) {
            //cerr << "Equivalent\n" << p << candidate.first << endl;

            if (candidate.first.size() > p.size()) {
                indexes_to_be_deleted.push_back(i);
                deleted_size += candidate.second.size() / EXAMPLE_NUM;
            } else {
                has_equivalent_program = true;
            }
        }
    }

    if (!has_equivalent_program) {
        candidates.push_back({p, examples});
        this->size += examples.size() / EXAMPLE_NUM;
    }

    for_each(indexes_to_be_deleted.rbegin(), indexes_to_be_deleted.rend(), [&candidates](const auto &i) {
        candidates.erase(candidates.begin() + i);
    });
    this->size -= deleted_size;
}

Dataset::Dataset() : size(0) {}

struct AsyncDataset {
    AsyncDataset() {
        this->size.store(0);
        this->abort.store(false);
    }

    future<DatasetForOneInputType> dataset;
    atomic<size_t> size;
    atomic<bool> abort;
};

experimental::optional<Dataset> generate_dataset(
        size_t min_length, size_t max_length, size_t dataset_size, size_t example_per_program) {
    auto functions = all_functions;
    functions.erase(find(functions.begin(), functions.end(), Function::ReadInt));
    functions.erase(find(functions.begin(), functions.end(), Function::ReadList));

    // Enumerate read_{list, int}
    Restriction r_for_read;
    r_for_read.min_length = 1;
    r_for_read.max_length = max_length;
    r_for_read.functions = { Function::ReadInt, Function::ReadList };

    Restriction r;
    r.min_length = min_length;
    r.max_length = max_length;
    r.functions = functions;
    r.predicates = all_predicate_lambdas;
    r.one_argument_lambda = all_one_argument_lambdas;
    r.two_arguments_lambda = all_two_arguments_lambdas;

    auto calc_info = [](const Program& p, const int &i) { return i; };

    vector<unique_ptr<AsyncDataset>> async_dataset;

    enumerate(
        r_for_read, calc_info,
        [&r, &calc_info, &dataset_size, &min_length, &max_length, &example_per_program, &async_dataset](const Program &p, const int &i) -> bool {
            r.min_length = min_length + p.size();
            r.max_length = max_length + p.size();
            async_dataset.push_back(make_unique<AsyncDataset>());
            auto &data = *async_dataset.back();
            auto id = async_dataset.size();
            data.dataset = std::async(std::launch::async,
                                      [r, calc_info, dataset_size, example_per_program, p, i, id, &data]() {
                                          DatasetForOneInputType d;
                                          enumerate(
                                                  r, calc_info,
                                                  [&dataset_size, &example_per_program, &d, &data, &id](const Program &p,
                                                                                                   const int &i) -> bool {
                                                      std::this_thread::yield();

                                                      // Check program
                                                      //// Unused program
                                                      if (has_unused_variable(p)) {
                                                          return true;
                                                      }

                                                      // Generate example
                                                      auto examples_ = generate_examples(p, example_per_program);

                                                      if (!examples_) {
                                                          cerr << "Fail to generate examples" << endl;
                                                          return true;
                                                      }

                                                      auto examples = examples_.value();
                                                      d.insert(p, examples);

                                                      //cerr << "Generating dataset... (" << id << ") " << d.size;
                                                      //if (dataset_size != 0) {
                                                      //    cerr << " / " << dataset_size;
                                                      //}
                                                      //cerr << endl;

                                                      data.size.store(d.size);

                                                      if (data.abort.load()) {
                                                          return false;
                                                      }

                                                      if (dataset_size == 0) {
                                                          return true;
                                                      } else {
                                                          return d.size < dataset_size;
                                                      }
                                                  },
                                                  p, i
                                          );


                                          return d;
                                      });
            return true;
        },
        0
    );

    // Run monitor thread
    atomic<bool> is_finished;
    is_finished = false;
    auto monitor = thread([&async_dataset, &is_finished, &dataset_size]() {
        while (true) {
            this_thread::sleep_for(chrono::seconds(30));
            if (is_finished) {
                return ;
            }

            size_t size = 0;
            for (auto &d: async_dataset) {
                size += d->size.load();
            }
            cerr << "Progress: " << size;
            if (dataset_size != 0) {
                cerr << " / " << dataset_size;
            }
            cerr << endl;
            if (dataset_size != 0 && size >= dataset_size) {
                // Finish all enumeration
                for (auto &d: async_dataset) {
                    d->abort.store(true);
                }
                return ;
            }
        }
    });

    // Wait for all futures
    Dataset dataset;
    for (auto &d: async_dataset) {
        auto x = d->dataset.get();
        dataset.programs.reserve(x.int_output_programs.size() + x.list_output_programs.size() + dataset.programs.size());
        for (auto &y: x.int_output_programs) {
            dataset.programs.push_back(y);
        }
        for (auto &y: x.list_output_programs) {
            dataset.programs.push_back(y);
        }
        dataset.size += x.size;
    }
    is_finished = true;
    monitor.join();

    return dataset;
}


experimental::optional<Dataset> generate_random_dataset(
        size_t min_length, size_t max_length, size_t dataset_size, size_t example_per_program) {
    auto functions = all_functions;
    functions.erase(find(functions.begin(), functions.end(), Function::ReadInt));
    functions.erase(find(functions.begin(), functions.end(), Function::ReadList));

    // Enumerate read_{list, int}
    Restriction r_for_read;
    r_for_read.min_length = 1;
    r_for_read.max_length = 3; // TODO
    r_for_read.functions = { Function::ReadInt, Function::ReadList };

    Restriction r;
    r.min_length = min_length;
    r.max_length = max_length;
    r.functions = functions;
    r.predicates = all_predicate_lambdas;
    r.one_argument_lambda = all_one_argument_lambdas;
    r.two_arguments_lambda = all_two_arguments_lambdas;

    auto calc_info = [](const Program& p, const int &i) { return i; };

    vector<unique_ptr<AsyncDataset>> async_dataset;

    enumerate(
            r_for_read, calc_info,
            [&r, &calc_info, &dataset_size, &min_length, &max_length, &example_per_program, &async_dataset](const Program &p, const int &i) -> bool {
                r.min_length = min_length;
                r.max_length = max_length;
                async_dataset.push_back(make_unique<AsyncDataset>());
                auto &data = *async_dataset.back();
                auto id = async_dataset.size();
                auto tenv = generate_type_environment(p).value();
                data.dataset = std::async(std::launch::async,
                                          [r, calc_info, dataset_size, example_per_program, tenv, p, i, id, &data]() {
                                              DatasetForOneInputType d;
                                              while (true) {
                                                  std::this_thread::yield();
                                                  if (data.abort.load()) {
                                                      break ;
                                                  }
                                                  try {
                                                      auto p2 = generate_random_program(r, p, tenv);
                                                      if (!p2) {
                                                          continue;
                                                      }
                                                      auto program = p2.value();

                                                      // Check program
                                                      //// Unused program
                                                      if (has_unused_variable(program)) {
                                                          continue;
                                                      }

                                                      // Generate example
                                                      auto examples_ = generate_examples(program, example_per_program);

                                                      if (!examples_) {
                                                          cerr << "Fail to generate examples" << endl;
                                                          continue;
                                                      }

                                                      auto examples = examples_.value();
                                                      d.insert(program, examples);

                                                      data.size.store(d.size);
                                                  } catch (std::exception e) {
                                                  }
                                              }
                                              return d;
                                          });
                return true;
            },
            0
    );

    // Run monitor thread
    atomic<bool> is_finished;
    is_finished = false;
    auto monitor = thread([&async_dataset, &is_finished, &dataset_size]() {
        size_t size_old = 0;
        while (true) {
            this_thread::sleep_for(chrono::seconds(30));
            if (is_finished) {
                return ;
            }

            size_t size = 0;
            for (auto &d: async_dataset) {
                size += d->size.load();
            }
            cerr << "Progress: " << size;
            if (dataset_size != 0) {
                cerr << " / " << dataset_size;
            }

            cerr << endl;
            if (size == size_old || (dataset_size != 0 && size >= dataset_size)) {
                // Finish all enumeration
                for (auto &d: async_dataset) {
                    d->abort.store(true);
                }
                return ;
            }
            size_old = size;
        }
    });

    // Wait for all futures
    Dataset dataset;
    for (auto &d: async_dataset) {
        auto x = d->dataset.get();
        dataset.programs.reserve(x.int_output_programs.size() + x.list_output_programs.size() + dataset.programs.size());
        for (auto &y: x.int_output_programs) {
            dataset.programs.push_back(y);
        }
        for (auto &y: x.list_output_programs) {
            dataset.programs.push_back(y);
        }
        dataset.size += x.size;
    }
    is_finished = true;
    monitor.join();

    return dataset;
}
