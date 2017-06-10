#include <gtest/gtest.h>
#include "dsl/ast.h"

using namespace dsl;

TEST(ArgumentTest, PredicateTest) {
    Argument pred(PredicateLambda::IsEven);

    EXPECT_TRUE(pred.predicate());
    EXPECT_EQ(PredicateLambda::IsEven, pred.predicate().value());

    Argument arg(0);
    EXPECT_FALSE(arg.predicate());
}

TEST(ArgumentTest, TwoArgumentsLambdaTest) {
    Argument minus(TwoArgumentsLambda::Minus);

    EXPECT_TRUE(minus.two_arguments_lambda());
    EXPECT_EQ(TwoArgumentsLambda::Minus, minus.two_arguments_lambda().value());

    Argument arg(0);
    EXPECT_FALSE(arg.two_arguments_lambda());
}


TEST(ArgumentTest, LambdaTest) {
    Argument lambda(OneArgumentLambda::Divide4);

    EXPECT_TRUE(lambda.one_argument_lambda());
    EXPECT_EQ(OneArgumentLambda::Divide4, lambda.one_argument_lambda().value());

    Argument arg(0);
    EXPECT_FALSE(arg.one_argument_lambda());
}

TEST(ArgumentTest, VariableTest) {
    Argument var(0);

    EXPECT_TRUE(var.variable());
    EXPECT_EQ(0, var.variable().value());

    Argument pred(PredicateLambda::IsNegative);
    EXPECT_FALSE(pred.variable());
}