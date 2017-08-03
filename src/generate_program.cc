#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include "dsl/utils.h"
#include "find-program.h"

using namespace std;
using namespace dsl;

Value to_value(const string &line) {
    istringstream iss(line);
    string type;
    iss >> type;
    vector<int> vec;
    while (!iss.eof()) {
        int x;
        iss >> x;
        vec.push_back(x);
    }

    if (type == "Integer") {
        return Value(vec.at(0));
    } else {
        return Value(vec);
    }
}
Input read_input() {
    string line;
    Input input;
    while (true) {
        getline(cin, line);

        if (line == "---") {
            return input;
        }
        input.push_back(to_value(line));
    }
}
Output read_output() {
    string line;
    getline(cin, line);
    Output output = to_value(line);
    getline(cin, line); // Read "---"

    return output;
}

int main(int argc, char **argv) {
    size_t max_length = 4;
    bool is_dfs = false;
    bool is_none = false;

    if (argc >= 2) {
        max_length = atoi(argv[1]);
    }
    if (argc >= 3) {
        auto x = string(argv[2]);
        if (x == "dfs") {
            is_dfs = true;
        } else if (x == "none") {
            is_none = true;
        }
    }

    // Read examples
    vector<Example> examples;
    while (cin) {
        auto input = read_input();
        if (input.size() == 0) {
            break;
        }
        auto output = read_output();
        examples.push_back({input, output});
    }

    // Read attribute
    vector<double> attr_;
    string tmp;
    cin >> tmp; // Read "Attribute:"
    while (cin) {
        double x;
        cin >> x;
        if (x < 0) {
            break;
        }
        attr_.push_back(x);
    }

    if (is_none) {
        for (auto &a: attr_) {
            a = 1.0;
        }
    } else {
        for (auto i = 0; i < all_functions.size() - 2; i++) {
            auto f = all_functions[i];
            cerr << stringify(f) << "\t";
        }
        for (auto x: all_predicate_lambdas) {
            cerr << stringify(x) << "\t";
        }
        for (auto x: all_one_argument_lambdas) {
            cerr << stringify(x) << "\t";
        }
        for (auto x: all_two_arguments_lambdas) {
            cerr << stringify(x) << "\t";
        }
        cerr << endl;

        for (const auto &a: attr_) {
            cerr << setprecision(3) << a << "\t";
        }
        cerr << endl;
    }

    // Find program
    auto p = (is_dfs)
             ? dfs(max_length, Attribute(attr_), examples)
             : (is_none)
             ? dfs(max_length, Attribute(attr_), examples)
             : sort_and_add(max_length, Attribute(attr_), examples);
    if (p) {
        cout << p.value() << endl;
        return 0;
    } else {
        cout << "not found" << endl;
        return 1;
    }
}