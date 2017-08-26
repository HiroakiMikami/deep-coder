#include <gtest/gtest.h>
#include <vector>
#include "dataset-generator.h"
#include "dsl/utils.h"

using namespace std;
using namespace dsl;

TEST(HasUnusedVariableTest, SimpleTest) {
    auto p1 = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::ReadInt, {})
    };
    EXPECT_TRUE(has_unused_variable(p1));

    auto p2 = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::ReadInt, {}),
            Statement(2, Function::Take, {1, 0})
    };
    EXPECT_FALSE(has_unused_variable(p2));
}

TEST(DatasetForOneInputTypeTest, InsertTest) {
    DatasetForOneInputType dataset;
    auto p1 = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::Minimum, {0})
    };
    auto p2 = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::Sort, {0}),
            Statement(2, Function::Head, {1})
    };
    auto p3 = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::Maximum, {0})
    };

    vector<Example> examples1 = {
            {{Value({0, 1})}, Value(0)},
            {{Value({-1, 10})}, Value(-1)},
            {{Value({-1, 10})}, Value(-1)},
            {{Value({-1, 10})}, Value(-1)},
            {{Value({-1, 10})}, Value(-1)}
    };

    vector<Example> examples2 = {
            {{Value({0, 1})}, Value(0)},
            {{Value({-1, 10})}, Value(-1)},
            {{Value({-1, 10})}, Value(-1)},
            {{Value({-1, 10})}, Value(-1)},
            {{Value({-1, 10})}, Value(-1)}
    };

    vector<Example> examples3 = {
            {{Value({0, 1})}, Value(1)},
            {{Value({-1, 10})}, Value(10)},
            {{Value({-1, 10})}, Value(10)},
            {{Value({-1, 10})}, Value(10)},
            {{Value({-1, 10})}, Value(10)}
    };
    dataset.insert(p2, examples2);
    EXPECT_EQ(1, dataset.int_output_programs.size());
    EXPECT_EQ(1, dataset.size);

    dataset.insert(p1, examples1);
    EXPECT_EQ(1, dataset.int_output_programs.size());
    EXPECT_EQ(1, dataset.size);

    dataset.insert(p3, examples3);
    EXPECT_EQ(2, dataset.int_output_programs.size());
    EXPECT_EQ(2, dataset.size);

    DatasetForOneInputType dataset2;
    dataset2.insert(p1, examples1);
    dataset2.insert(p2, examples2);
    EXPECT_EQ(1, dataset2.int_output_programs.size());
    EXPECT_EQ(1, dataset2.size);
}

TEST(GenerateDatasetTest, SimpleTest) {
    auto x = generate_dataset(1, 2, 20);
    EXPECT_TRUE(static_cast<bool>(x));

    EXPECT_TRUE(x.value().size >= 10);

    for (auto &i: x.value().programs) {
        cout << i.first;
        for (auto &y: i.second) {
            cout << y.input << y.output << endl;
        }
    }
}