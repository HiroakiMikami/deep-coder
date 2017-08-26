#include <gtest/gtest.h>
#include "find-program.h"

using namespace std;
using namespace dsl;

TEST(DfsTest, FindProgramTest1) {
    vector<Example> examples = {
            {{Value({2, 1, 5})}, Value(1)},
            {{Value({-1, 1, 5})}, Value(-1)}
    };

    Attribute attr({Statement(0, Function::ReadList, {}), Statement(1, Function::Minimum, {0})});
    auto p = dfs(2, attr, examples);
    EXPECT_TRUE(static_cast<bool>(p));
    EXPECT_EQ(2, p.value().size());
    EXPECT_EQ(1, p.value()[1].variable);
    EXPECT_EQ(Function::Minimum, p.value()[1].function);
    EXPECT_EQ(0, p.value()[1].arguments[0].variable().value());
}
TEST(DfsTest, FindProgramTest2) {
    vector<Example> examples = {
            {{Value({2, 1, 5})}, Value({4, 1, 25})},
            {{Value({-1, 1, 5})}, Value({1, 1, 25})}
    };

    Attribute attr({
                           Statement(0, Function::ReadList, {}),
                           Statement(1, Function::Map, {OneArgumentLambda::Pow2, 0})
                   });
    auto p = dfs(2, attr, examples);
    EXPECT_TRUE(static_cast<bool>(p));
    EXPECT_EQ(2, p.value().size());
    EXPECT_EQ(1, p.value()[1].variable);
    EXPECT_EQ(Function::Map, p.value()[1].function);
    EXPECT_EQ(OneArgumentLambda::Pow2, p.value()[1].arguments[0].one_argument_lambda().value());
    EXPECT_EQ(0, p.value()[1].arguments[1].variable().value());
}
TEST(DfsTest, FindProgramTest3) {
    vector<Example> examples = {
            {{Value({2, 1, 5})}, Value(25)},
            {{Value({-1, 1, 5})}, Value(25)}
    };

    Attribute attr({
                           Statement(0, Function::ReadList, {}),
                           Statement(1, Function::Map, {OneArgumentLambda::Pow2, 0}),
                           Statement(2, Function::Maximum, {1})
                   });
    auto p = dfs(2, attr, examples);
    EXPECT_TRUE(static_cast<bool>(p));
    EXPECT_EQ(3, p.value().size());
    EXPECT_EQ(1, p.value()[1].variable);
    EXPECT_EQ(Function::Map, p.value()[1].function);
    EXPECT_EQ(OneArgumentLambda::Pow2, p.value()[1].arguments[0].one_argument_lambda().value());
    EXPECT_EQ(0, p.value()[1].arguments[1].variable().value());

    EXPECT_EQ(2, p.value()[2].variable);
    EXPECT_EQ(Function::Maximum, p.value()[2].function);
    EXPECT_EQ(1, p.value()[2].arguments[0].variable().value());
}

TEST(SortAndAddTest, FindProgramTest1) {
    vector<Example> examples = {
            {{Value({2, 1, 5})}, Value(1)},
            {{Value({-1, 1, 5})}, Value(-1)}
    };

    Attribute attr({Statement(0, Function::ReadList, {}), Statement(1, Function::Minimum, {0})});
    auto p = sort_and_add(2, attr, examples);
    EXPECT_TRUE(static_cast<bool>(p));
    EXPECT_EQ(2, p.value().size());
    EXPECT_EQ(1, p.value()[1].variable);
    EXPECT_EQ(Function::Minimum, p.value()[1].function);
    EXPECT_EQ(0, p.value()[1].arguments[0].variable().value());
}
TEST(SortAndAddTest, FindProgramTest2) {
    vector<Example> examples = {
            {{Value({2, 1, 5})}, Value({4, 1, 25})},
            {{Value({-1, 1, 5})}, Value({1, 1, 25})}
    };

    Attribute attr({
                           Statement(0, Function::ReadList, {}),
                           Statement(1, Function::Map, {OneArgumentLambda::Pow2, 0})
                   });
    auto p = sort_and_add(2, attr, examples);
    EXPECT_TRUE(static_cast<bool>(p));
    EXPECT_EQ(2, p.value().size());
    EXPECT_EQ(1, p.value()[1].variable);
    EXPECT_EQ(Function::Map, p.value()[1].function);
    EXPECT_EQ(OneArgumentLambda::Pow2, p.value()[1].arguments[0].one_argument_lambda().value());
    EXPECT_EQ(0, p.value()[1].arguments[1].variable().value());
}
TEST(SortAndAddTest, FindProgramTest3) {
    vector<Example> examples = {
            {{Value({2, 1, 5})}, Value(25)},
            {{Value({-1, 1, 5})}, Value(25)}
    };

    Attribute attr({
                           Statement(0, Function::ReadList, {}),
                           Statement(1, Function::Map, {OneArgumentLambda::Pow2, 0}),
                           Statement(2, Function::Maximum, {1})
                   });
    auto p = sort_and_add(2, attr, examples);
    EXPECT_TRUE(static_cast<bool>(p));
    EXPECT_EQ(3, p.value().size());
    EXPECT_EQ(1, p.value()[1].variable);
    EXPECT_EQ(Function::Map, p.value()[1].function);
    EXPECT_EQ(OneArgumentLambda::Pow2, p.value()[1].arguments[0].one_argument_lambda().value());
    EXPECT_EQ(0, p.value()[1].arguments[1].variable().value());

    EXPECT_EQ(2, p.value()[2].variable);
    EXPECT_EQ(Function::Maximum, p.value()[2].function);
    EXPECT_EQ(1, p.value()[2].arguments[0].variable().value());
}