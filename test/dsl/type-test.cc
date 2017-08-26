#include <gtest/gtest.h>
#include "dsl/type.h"

using namespace std;
using namespace dsl;

TEST(DslTypeTest, CheckTest) {
    TypeEnvironment e0;
    auto e1 = check(Statement(0, Function::ReadInt, {}), e0);

    EXPECT_TRUE(static_cast<bool>(e1));
    EXPECT_EQ(1, e1.value().size());
    EXPECT_EQ(Type::Integer, e1.value().find(0)->second);

    auto e2 = check(Statement(1, Function::Head, {Argument(0)}), e1.value());
    EXPECT_FALSE(static_cast<bool>(e2));

    auto e3 = check(Statement(0, Function::ReadInt, {}), e1.value());
    EXPECT_FALSE(static_cast<bool>(e3));
}

TEST(DslTypeTest, GenerateTypeEnvironmentTest) {
    Program p1 = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::ReadList, {}),
            Statement(2, Function::ZipWith, {TwoArgumentsLambda::Minus, 0, 1}),
            Statement(3, Function::Count, {PredicateLambda::IsPositive, 2})
    };

    auto tenv1 = generate_type_environment(p1);
    EXPECT_TRUE(static_cast<bool>(tenv1));
    EXPECT_EQ(4, tenv1.value().size());
    EXPECT_EQ(Type::List, tenv1.value().find(0)->second);
    EXPECT_EQ(Type::List, tenv1.value().find(1)->second);
    EXPECT_EQ(Type::List, tenv1.value().find(2)->second);
    EXPECT_EQ(Type::Integer, tenv1.value().find(3)->second);

    Program p2 = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::ReadInt, {}),
            Statement(2, Function::ZipWith, {TwoArgumentsLambda::Minus, 0, 1}),
            Statement(3, Function::Count, {PredicateLambda::IsPositive, 2})
    };
    EXPECT_FALSE(static_cast<bool>(generate_type_environment(p2)));
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