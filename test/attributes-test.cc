#include <gtest/gtest.h>
#include "attribute.h"

using namespace std;
using namespace dsl;

TEST(AttributeTest, ConstructorTest) {
    Program p = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::Head, {0}),
            Statement(2, Function::Map, {OneArgumentLambda::Plus1, 0}),
            Statement(3, Function::Count, {PredicateLambda::IsPositive, 0})
    };

    Attribute a(p);
    EXPECT_EQ(1, a.function_presence[Function::Head]);
    EXPECT_EQ(0, a.function_presence[Function::Last]);
    EXPECT_EQ(1, a.function_presence[Function::Map]);
    EXPECT_EQ(0, a.function_presence[Function::Filter]);
    EXPECT_EQ(1, a.function_presence[Function::Count]);
    EXPECT_TRUE(a.function_presence.find(Function::ReadList) == a.function_presence.end());

    EXPECT_EQ(1, a.one_argument_lambda_presence[OneArgumentLambda::Plus1]);
    EXPECT_EQ(0, a.one_argument_lambda_presence[OneArgumentLambda::Minus1]);

    EXPECT_EQ(1, a.predicate_presence[PredicateLambda::IsPositive]);
    EXPECT_EQ(0, a.predicate_presence[PredicateLambda::IsNegative]);
}

TEST(AttributeTest, ConstructorTest2) {
    vector<double> v(34, 0.0);
    v[0] = 1.0;
    Attribute a(v);
    EXPECT_EQ(1, a.function_presence[Function::Head]);
    EXPECT_EQ(0, a.function_presence[Function::Last]);
    EXPECT_EQ(0, a.function_presence[Function::Map]);
    EXPECT_EQ(0, a.function_presence[Function::Filter]);
    EXPECT_EQ(0, a.function_presence[Function::Count]);
    EXPECT_TRUE(a.function_presence.find(Function::ReadList) == a.function_presence.end());

    EXPECT_EQ(0, a.one_argument_lambda_presence[OneArgumentLambda::Plus1]);
    EXPECT_EQ(0, a.one_argument_lambda_presence[OneArgumentLambda::Minus1]);

    EXPECT_EQ(0, a.predicate_presence[PredicateLambda::IsPositive]);
    EXPECT_EQ(0, a.predicate_presence[PredicateLambda::IsNegative]);
}

TEST(AttributeTest, SerializeTest) {
    Program p = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::Head, {0}),
            Statement(2, Function::Map, {OneArgumentLambda::Plus1, 0}),
            Statement(3, Function::Count, {PredicateLambda::IsPositive, 0})
    };

    Attribute a(p);
    std::vector<double> x(a);

    EXPECT_EQ(1, x[0]);
    EXPECT_EQ(0, x[1]);
    EXPECT_EQ(1, x[all_functions.size()]);
}

TEST(AttributeTest, DeserializeTest) {

}