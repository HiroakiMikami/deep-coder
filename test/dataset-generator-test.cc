#include <gtest/gtest.h>
#include <vector>
#include "dataset-generator.h"

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

TEST(DatasetTest, InsertTest) {
    Dataset dataset;
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
    auto p4 = {
            Statement(0, Function::ReadInt, {}),
            Statement(1, Function::ReadList, {}),
            Statement(2, Function::Take, {0, 1})
    };

    vector<Example> examples1 = {
            {{Value({0, 1})}, Value(0)},
            {{Value({-1, 10})}, Value(-1)}
    };

    vector<Example> examples2 = {
            {{Value({0, 1})}, Value(0)},
            {{Value({-1, 10})}, Value(-1)}
    };

    vector<Example> examples3 = {
            {{Value({0, 1})}, Value(1)},
            {{Value({-1, 10})}, Value(10)}
    };

    vector<Example> examples4 = {
            {{Value(2), Value({0, 1, 2})}, Value({0, 1})}
    };

    dataset.insert(p2, examples2);
    EXPECT_EQ(1, dataset.programs.size());
    EXPECT_EQ(1, dataset.size);

    dataset.insert(p1, examples1);
    EXPECT_EQ(1, dataset.programs.size());
    EXPECT_EQ(1, dataset.size);
    for (auto &x: dataset.programs) {
        for (auto &y: x.second) {
            EXPECT_EQ(2, y.first.size());
        }
    }

    dataset.insert(p3, examples3);
    EXPECT_EQ(1, dataset.programs.size());
    EXPECT_EQ(2, dataset.size);

    dataset.insert(p4, examples4);
    EXPECT_EQ(2, dataset.programs.size());
    EXPECT_EQ(3, dataset.size);


    Dataset dataset2;
    dataset2.insert(p1, examples1);
    dataset2.insert(p2, examples2);
    EXPECT_EQ(1, dataset2.programs.size());
    EXPECT_EQ(1, dataset2.size);
    for (auto &x: dataset2.programs) {
        for (auto &y: x.second) {
            EXPECT_EQ(2, y.first.size());
        }
    }
}