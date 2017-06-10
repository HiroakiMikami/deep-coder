#include <gtest/gtest.h>
#include "dsl/type.h"

using namespace std;
using namespace dsl;

TEST(DslTypeTest, CheckTest) {
    TypeEnvironment e0;
    auto e1 = check(Statement(0, Function::ReadInt, {}), e0);

    EXPECT_TRUE(e1);
    EXPECT_EQ(1, e1.value().size());
    EXPECT_EQ(Type::Integer, e1.value().find(0)->second);

    auto e2 = check(Statement(1, Function::Head, {Argument(0)}), e1.value());
    EXPECT_FALSE(e2);

    auto e3 = check(Statement(0, Function::ReadInt, {}), e1.value());
    EXPECT_FALSE(e3);
}

TEST(DslTypeTest, IsValidTest) {
    Program p1 = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::ReadList, {}),
            Statement(2, Function::ZipWith, {TwoArgumentsLambda::Minus, 0, 1}),
            Statement(3, Function::Count, {PredicateLambda::IsPositive, 2})
    };
    EXPECT_TRUE(is_valid(p1));

    Program p2 = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::ReadInt, {}),
            Statement(2, Function::ZipWith, {TwoArgumentsLambda::Minus, 0, 1}),
            Statement(3, Function::Count, {PredicateLambda::IsPositive, 2})
    };
    EXPECT_FALSE(is_valid(p2));
}