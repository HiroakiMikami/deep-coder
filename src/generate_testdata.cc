#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <algorithm>
#include <string>
#include <tuple>
#include "dsl/utils.h"
#include "dataset-generator.h"
#include "attribute.h"

using namespace std;
using namespace dsl;

void output_value(ofstream &stream, const Value &value) {
    if (value.integer()) {
        stream << value.integer().value();
    } else {
        auto l = value.list().value();
        stream << "[";
        for (auto i = 0; i < l.size(); i++) {
            auto x = l.at(i);

            stream << x;
            if (i != (l.size() - 1)) {
                stream << ",";
            }
        }
        stream << "]";
    }
}
void output_input(ofstream &stream, const Input &input) {
    stream << "[";
    for (auto i = 0; i < input.size(); i++) {
        auto x = input.at(i);

        output_value(stream, x);
        if (i != (input.size() - 1)) {
            stream << ",";
        }
    }
    stream << "]";
}

void output_attribute(const Attribute &attr) {
    std::vector<double> vec = attr;
    cout << "[";
    for (auto i = 0; i < vec.size(); i++) {
        cout << vec.at(i);
        if (i != (vec.size() - 1)) {
            cout << ",";
        }
    }
    cout << "]";
}

int main(int argc, char **argv) {
    size_t max_length = 4;
    size_t dataset_size = 100;

    if (argc < 2) {
        cout << "no-dir" << endl;
        return 1;
    }
    auto dir = argv[1];

    if (argc >= 3) {
        max_length = atoi(argv[2]);
    }
    if (argc >= 4) {
        dataset_size = atoi(argv[3]);
    }

    auto dataset_per_length = dataset_size / max_length;

    auto functions = all_functions;
    functions.erase(find(functions.begin(), functions.end(), Function::ReadInt));
    functions.erase(find(functions.begin(), functions.end(), Function::ReadList));

    // Generate dataset
    cerr << "Generate Dataset " << max_length << " " << max_length * dataset_size * 5 << endl;
    auto dataset_opt = generate_random_dataset(1, max_length, max_length * dataset_size * 5, EXAMPLE_NUM);
    if (!dataset_opt) {
        return 1;
    }

    cerr << "Extract examples" << endl;
    auto dataset = dataset_opt.value();

    vector<int> testdata_num(max_length + 1, 0);
    for (auto i = 1; i < max_length; i++) {
        testdata_num[i] = 10;
    }
    testdata_num[max_length] = dataset_size - (10 * (max_length - 1));

    vector<vector<pair<Program, vector<Example>>>> testdata(max_length + 1);
    cerr << dataset.programs.size() << endl;
    static std::random_device rnd;
    static std::mt19937 mt(rnd());
    std::uniform_int_distribution<> examples(0, dataset.programs.size() - 1);
    while (true) {
        auto i = examples(mt);

        cerr << "Select : " << i << endl;
        auto x = dataset.programs.at(i);
        auto p = x.first;
        auto e = x.second;

        cerr << p << endl;
        auto len = 0;
        for (auto &s: p) {
            if (s.function == Function::ReadInt || s.function == Function::ReadList) {
                continue ;
            }
            len += 1;
        }


        if (testdata[len].size() < testdata_num[len]) {
            testdata[len].push_back(x);
        }

        bool f = true;
        for (auto j = 0; j < testdata.size(); j++) {
            if (testdata.at(j).size() < testdata_num.at(j)) {
                f = false;
                break;
            }
        }
        if (f) {
            break ;
        }
    }

    int cnt = 0;
    for (auto l = 0; l < testdata.size(); l++) {
        for (auto x: testdata[l]) {
            auto p = x.first;
            auto e = x.second;
            stringstream pfile;
            pfile << dir << "/" << cnt << "-program";
            ofstream ofs(pfile.str());
            ofs << p;

            stringstream efile;
            efile << dir << "/" << cnt << "-example";
            ofstream ofs2(efile.str());
            ofs2 << "[\n";
            for (auto j = 0; j < e.size(); j++) {
                ofs2 << "{\"input\":";
                output_input(ofs2, e[j].input);
                ofs2<< ",\"output\":";
                output_value(ofs2, e[j].output);
                ofs2 << "}";
                if (j != (e.size() - 1)) {
                    ofs2 << ",";
                }
                ofs2 << "\n";
            }
            ofs2 << "]";

            cnt += 1;
        }
    }

    return 0;
}