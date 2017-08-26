#include <vector>
#include <gtest/gtest.h>
#include "dsl/interpreter.h"

using namespace std;
using namespace dsl;

TEST(ValueTest, IntegerValueTest) {
    Value v = 0;
    EXPECT_EQ(0, v.integer().value());
    EXPECT_FALSE(static_cast<bool>(v.list()));
    EXPECT_FALSE(v.is_null());
}
TEST(ValueTest, ListValueTest) {
    Value v = std::vector<int>();
    EXPECT_EQ(std::vector<int>(), v.list().value());
    EXPECT_FALSE(static_cast<bool>(v.integer()));
    EXPECT_FALSE(v.is_null());
}
TEST(ValueTest, NullValueTest) {
    Value v;
    EXPECT_FALSE(static_cast<bool>(v.integer()));
    EXPECT_FALSE(static_cast<bool>(v.list()));
    EXPECT_TRUE(v.is_null());
}
TEST(ValueTest, EqualsTest) {
    Value v1(0);
    Value v2(0);
    Value v3(1);
    Value v4({0, 1});
    Value v5({0, 1});
    Value v6({0, 1, 2});

    EXPECT_TRUE(v1 == v2);
    EXPECT_FALSE(v1 == v3);
    EXPECT_FALSE(v1 == v4);
    EXPECT_TRUE(v4 == v5);
    EXPECT_FALSE(v4 == v6);
}

TEST(ProceedTest, EvalHeadTest) {
    Environment e0({{0, Value({0, 1, 2})}, {1, Value(vector<int>(0))}}, {});

    auto e1 = proceed(Statement(2, Function::Head, {0}), e0);
    EXPECT_TRUE(static_cast<bool>(e1));
    EXPECT_EQ(Value(0), e1.value().variables.find(2)->second);

    auto e2 = proceed(Statement(2, Function::Head, {1}), e0);
    EXPECT_TRUE(static_cast<bool>(e2));
    EXPECT_EQ(Value(), e2.value().variables.find(2)->second);
}
TEST(ProceedTest, EvalLastTest) {
    Environment e0({{0, Value({0, 1, 2})}, {1, Value(vector<int>(0))}}, {});

    auto e1 = proceed(Statement(2, Function::Last, {0}), e0);
    EXPECT_TRUE(static_cast<bool>(e1));
    EXPECT_EQ(Value(2), e1.value().variables.find(2)->second);

    auto e2 = proceed(Statement(2, Function::Last, {1}), e0);
    EXPECT_TRUE(static_cast<bool>(e2));
    EXPECT_EQ(Value(), e2.value().variables.find(2)->second);
}

TEST(ProceedTest, EvalTakeTest) {
    Environment e0({{0, Value({0, 1, 2})}, {1, Value(1)}, {2, Value(5)}, {4, Value(-1)}}, {});

    auto e1 = proceed(Statement(3, Function::Take, {1, 0}), e0);
    EXPECT_TRUE(static_cast<bool>(e1));
    EXPECT_EQ(Value(vector<int>(1, 0)), e1.value().variables.find(3)->second);

    auto e2 = proceed(Statement(3, Function::Take, {2, 0}), e0);
    EXPECT_TRUE(static_cast<bool>(e2));
    EXPECT_EQ(Value({0, 1, 2}), e2.value().variables.find(3)->second);

    auto e3 = proceed(Statement(3, Function::Take, {4, 0}), e0);
    EXPECT_TRUE(static_cast<bool>(e3));
    EXPECT_EQ(Value(vector<int>()), e3.value().variables.find(3)->second);
}
TEST(ProceedTest, EvalDropTest) {
    Environment e0({{0, Value({0, 1, 2})}, {1, Value(1)}, {2, Value(5)}}, {});

    auto e1 = proceed(Statement(3, Function::Drop, {1, 0}), e0);
    EXPECT_TRUE(static_cast<bool>(e1));
    EXPECT_EQ(Value({1, 2}), e1.value().variables.find(3)->second);

    auto e2 = proceed(Statement(3, Function::Drop, {2, 0}), e0);
    EXPECT_TRUE(static_cast<bool>(e2));
    EXPECT_EQ(Value(vector<int>(0)), e2.value().variables.find(3)->second);
}

TEST(ProceedTest, EvalAccessTest) {
    Environment e0({{0, Value({0, 1, 2})}, {1, Value(0)}, {2, Value(-1)}}, {});

    auto e1 = proceed(Statement(3, Function::Access, {1, 0}), e0);
    EXPECT_TRUE(static_cast<bool>(e1));
    EXPECT_EQ(Value(0), e1.value().variables.find(3)->second);

    auto e2 = proceed(Statement(3, Function::Last, {1, 2}), e0);
    EXPECT_TRUE(static_cast<bool>(e2));
    EXPECT_EQ(Value(), e2.value().variables.find(3)->second);
}

TEST(ProceedTest, EvalMinimumTest) {
    Environment e0({{0, Value({-1, 1, 2})}, {1, Value(vector<int>(0))}}, {});

    auto e1 = proceed(Statement(2, Function::Minimum, {0}), e0);
    EXPECT_TRUE(static_cast<bool>(e1));
    EXPECT_EQ(Value(-1), e1.value().variables.find(2)->second);

    auto e2 = proceed(Statement(2, Function::Minimum, {1}), e0);
    EXPECT_TRUE(static_cast<bool>(e2));
    EXPECT_EQ(Value(), e2.value().variables.find(2)->second);
}
TEST(ProceedTest, EvalMaximumTest) {
    Environment e0({{0, Value({-1, 1, 2})}, {1, Value(vector<int>(0))}}, {});

    auto e1 = proceed(Statement(2, Function::Maximum, {0}), e0);
    EXPECT_TRUE(static_cast<bool>(e1));
    EXPECT_EQ(Value(2), e1.value().variables.find(2)->second);

    auto e2 = proceed(Statement(2, Function::Maximum, {1}), e0);
    EXPECT_TRUE(static_cast<bool>(e2));
    EXPECT_EQ(Value(), e2.value().variables.find(2)->second);
}

TEST(ProceedTest, EvalReverseTest) {
    Environment e0({{0, Value({-1, 1, 2})}}, {});

    auto e1 = proceed(Statement(2, Function::Reverse, {0}), e0);
    EXPECT_TRUE(static_cast<bool>(e1));
    EXPECT_EQ(Value({2, 1, -1}), e1.value().variables.find(2)->second);
    EXPECT_EQ(Value({-1, 1, 2}), e1.value().variables.find(0)->second);
}
TEST(ProceedTest, EvalSortTest) {
    Environment e0({{0, Value({2, -1, 1})}, {1, Value(vector<int>(0))}}, {});

    auto e1 = proceed(Statement(2, Function::Sum, {0}), e0);
    EXPECT_TRUE(static_cast<bool>(e1));
    EXPECT_EQ(Value(2), e1.value().variables.find(2)->second);

    auto e2 = proceed(Statement(2, Function::Sum, {1}), e0);
    EXPECT_TRUE(static_cast<bool>(e2));
    EXPECT_EQ(Value(0), e2.value().variables.find(2)->second);
}

TEST(ProceedTest, EvalMapTest) {
    Environment e0({{0, Value({2, -1, 1})}}, {});

    auto e1 = proceed(Statement(1, Function::Map, {OneArgumentLambda::Pow2, 0}), e0);
    EXPECT_TRUE(static_cast<bool>(e1));
    EXPECT_EQ(Value({4, 1, 1}), e1.value().variables.find(1)->second);
}

TEST(ProceedTest, EvalFilterTest) {
    Environment e0({{0, Value({2, -1, 1})}}, {});

    auto e1 = proceed(Statement(1, Function::Filter, {PredicateLambda::IsPositive, 0}), e0);
    EXPECT_TRUE(static_cast<bool>(e1));
    EXPECT_EQ(Value({2, 1}), e1.value().variables.find(1)->second);
}
TEST(ProceedTest, EvalCountTest) {
    Environment e0({{0, Value({2, -1, 1})}}, {});

    auto e1 = proceed(Statement(1, Function::Count, {PredicateLambda::IsPositive, 0}), e0);
    EXPECT_TRUE(static_cast<bool>(e1));
    EXPECT_EQ(Value(2), e1.value().variables.find(1)->second);
}

TEST(ProceedTest, EvalZipWithTest) {
    Environment e0({{0, Value({2, -1, 1})}, {1, Value({3, 4})}}, {});

    auto e1 = proceed(Statement(2, Function::ZipWith, {TwoArgumentsLambda::Multiply, 0, 1}), e0);
    EXPECT_TRUE(static_cast<bool>(e1));
    EXPECT_EQ(Value({6, -4}), e1.value().variables.find(2)->second);
}
TEST(ProceedTest, EvalScanl1Test) {
    Environment e0({{0, Value({2, -1, 3})}, {1, Value(vector<int>(0))}}, {});

    auto e1 = proceed(Statement(2, Function::Scanl1, {TwoArgumentsLambda::Multiply, 0}), e0);
    EXPECT_TRUE(static_cast<bool>(e1));
    EXPECT_EQ(Value({2, -2, -6}), e1.value().variables.find(2)->second);

    auto e2 = proceed(Statement(2, Function::Scanl1, {TwoArgumentsLambda::Multiply, 1}), e0);
    EXPECT_TRUE(static_cast<bool>(e2));
    EXPECT_EQ(Value(vector<int>(0)), e2.value().variables.find(2)->second);
}

TEST(EvalTest, Program0Test) {
    auto p = {
            Statement(0, Function::ReadInt, {}),
            Statement(1, Function::ReadList, {}),
            Statement(2, Function::Sort, {1}),
            Statement(3, Function::Take, {0, 2}),
            Statement(4, Function::Sum, {3})
    };

    auto x = eval(p, {Value(2), Value({3, 5, 4, 7, 5})});
    EXPECT_TRUE(static_cast<bool>(x));
    EXPECT_EQ(Value(7), x.value());
}
TEST(EvalTest, Program1Test) {
    auto p = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::ReadList, {}),
            Statement(2, Function::Map, {OneArgumentLambda::Multiply3, 0}),
            Statement(3, Function::ZipWith, {TwoArgumentsLambda::Plus, 1, 2}),
            Statement(4, Function::Maximum, {3})
    };

    auto x = eval(p, {Value({6, 2, 4, 7, 9}), Value({5, 3, 6, 1, 0})});
    EXPECT_TRUE(static_cast<bool>(x));
    EXPECT_EQ(Value(27), x.value());
}
TEST(EvalTest, Program2Test) {
    auto p = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::ReadList, {}),
            Statement(2, Function::ZipWith, {TwoArgumentsLambda::Minus, 0, 1}),
            Statement(3, Function::Count, {PredicateLambda::IsPositive, 2})
    };

    auto x = eval(p, {Value({6, 2, 4, 7, 9}), Value({5, 3, 2, 1, 0})});
    EXPECT_TRUE(static_cast<bool>(x));
    EXPECT_EQ(Value(4), x.value());
}
TEST(EvalTest, Program3Test) {
    auto p = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::Scanl1, {TwoArgumentsLambda::Min, 0}),
            Statement(2, Function::ZipWith, {TwoArgumentsLambda::Minus, 0, 1}),
            Statement(3, Function::Filter, {PredicateLambda::IsPositive, 2}),
            Statement(4, Function::Sum, {3})
    };

    auto x = eval(p, {Value({8, 5, 7, 2, 5})});
    EXPECT_TRUE(static_cast<bool>(x));
    EXPECT_EQ(Value(5), x.value());
}
TEST(EvalTest, Program4Test) {
    auto p = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::ReadList, {}),
            Statement(2, Function::Sort, {0}),
            Statement(3, Function::Sort, {1}),
            Statement(4, Function::Reverse, {3}),
            Statement(5, Function::ZipWith, {TwoArgumentsLambda::Multiply, 2, 4}),
            Statement(6, Function::Sum, {5})
    };

    auto x = eval(p, {Value({7, 3, 8, 2, 5}), Value({2, 8, 9, 1, 3})});
    EXPECT_TRUE(static_cast<bool>(x));
    EXPECT_EQ(Value(79), x.value());
}
TEST(EvalTest, Program5Test) {
    auto p = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::Reverse, {0}),
            Statement(2, Function::ZipWith, {TwoArgumentsLambda::Min, 0, 1}),
    };

    auto x = eval(p, {Value({3, 7, 5, 2, 8})});
    EXPECT_TRUE(static_cast<bool>(x));
    EXPECT_EQ(Value({3, 2, 5, 2, 3}), x.value());
}
TEST(EvalTest, Program6Test) {
    auto p = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::ReadList, {}),
            Statement(2, Function::Map, {OneArgumentLambda::Minus1, 0}),
            Statement(3, Function::Map, {OneArgumentLambda::Minus1, 1}),
            Statement(4, Function::ZipWith, {TwoArgumentsLambda::Plus, 2, 3}),
            Statement(5, Function::Minimum, {4})
    };

    auto x = eval(p, {Value({4, 8, 11, 2}), Value({2, 3, 4, 1})});
    EXPECT_TRUE(static_cast<bool>(x));
    EXPECT_EQ(Value(1), x.value());
}
TEST(EvalTest, Program7Test) {
    auto p = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::ReadList, {}),
            Statement(2, Function::Scanl1, {TwoArgumentsLambda::Plus, 1}),
            Statement(3, Function::ZipWith, {TwoArgumentsLambda::Multiply, 0, 2}),
            Statement(4, Function::Sum, {3})
    };

    auto x = eval(p, {Value({4, 7, 2, 3}), Value({2, 1, 3, 1})});
    EXPECT_TRUE(static_cast<bool>(x));
    EXPECT_EQ(Value(62), x.value());
}
TEST(EvalTest, Program8Test) {
    auto p = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::Reverse, {0}),
            Statement(2, Function::ZipWith, {TwoArgumentsLambda::Minus, 1, 0}),
            Statement(3, Function::Filter, {PredicateLambda::IsPositive, 2}),
            Statement(4, Function::Sum, {3})
    };

    auto x = eval(p, {Value({1, 2, 4, 5, 7})});
    EXPECT_TRUE(static_cast<bool>(x));
    EXPECT_EQ(Value(9), x.value());
}
