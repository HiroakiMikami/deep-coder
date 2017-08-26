#include <iostream>
#include <gtest/gtest.h>
#include "example-generator.h"

using namespace std;
using namespace dsl;

TEST(GenerateIntegerTest, MinMaxTest) {
    IntegerConstraint c;
    c.min = -10;
    c.max = 20;

    for (auto i = 0; i < 100; i++) {
        auto x = generate_integer(c);
        EXPECT_TRUE(x >= -10);
        EXPECT_TRUE(x <= 20);
    }

    c.max = -20;
    auto x = generate_integer(c);
    EXPECT_FALSE(static_cast<bool>(x));
}

TEST(GenerateIntegerTest, SignTest) {
    IntegerConstraint c;
    c.min = -10;
    c.max = 20;

    c.sign = Sign::Positive;
    for (auto i = 0; i < 100; i++) {
        auto x = generate_integer(c);
        EXPECT_TRUE(x >= 1);
        EXPECT_TRUE(x <= 20);
    }

    c.sign = Sign::Negative;
    for (auto i = 0; i < 100; i++) {
        auto x = generate_integer(c);
        EXPECT_TRUE(x >= -10);
        EXPECT_TRUE(x <= -1);
    }

    c.sign = Sign::Zero;
    for (auto i = 0; i < 100; i++) {
        auto x = generate_integer(c);
        EXPECT_EQ(0, x);
    }

    c.sign = Sign::Positive;
    c.min = -100;
    c.max = -1;
    auto x = generate_integer(c);
    EXPECT_FALSE(static_cast<bool>(x));
}

TEST(GenerateIntegerTest, IsEvenTest) {
    IntegerConstraint c;
    c.min = -10;
    c.max = 20;

    c.is_even = true;
    for (auto i = 0; i < 100; i++) {
        auto x = generate_integer(c);
        EXPECT_TRUE((x.value() % 2) == 0);
        EXPECT_TRUE(x >= -10);
        EXPECT_TRUE(x <= 20);
    }

    c.is_even = false;
    for (auto i = 0; i < 100; i++) {
        auto x = generate_integer(c);
        EXPECT_EQ(1, abs(x.value() % 2));
        EXPECT_TRUE(x >= -10);
        EXPECT_TRUE(x <= 20);
    }
}

TEST(GenerateListTest, MinMaxLengthTest) {
    ListConstraint c;
    c.min_length = 10;
    c.max_length = 20;

    for (auto i = 0; i < 100; i++) {
        auto x = generate_list(c);
        EXPECT_TRUE(static_cast<bool>(x));
        EXPECT_TRUE(x.value().size() >= 10);
        EXPECT_TRUE(x.value().size() <= 20);
    }

    c.max_length = 5;
    auto x = generate_list(c);
    EXPECT_FALSE(static_cast<bool>(x));
}

TEST(GenerateListTest, IntegerConstraintTest) {
    ListConstraint c;
    c.min_length = 1;

    c.min = 5;
    c.max = 10;
    c.is_even = {true};

    for (auto i = 0; i < 100; i++) {
        auto x = generate_list(c);
        EXPECT_TRUE(static_cast<bool>(x));
        for (const auto &i: x.value()) {
            EXPECT_TRUE(i >= 5);
            EXPECT_TRUE(i <= 10);
            EXPECT_TRUE((i % 2) == 0);
        }
    }

    c.min = -1;
    c.max = 0;
    c.sign = {Sign::Positive};

    auto x = generate_list(c);
    EXPECT_FALSE(static_cast<bool>(x));
}

TEST(AnalyzeTest, SimpleAnalyzeTest) {
    auto p = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::ReadInt, {}),
            Statement(2, Function::Map, {OneArgumentLambda::Plus1, 0}),
            Statement(3, Function::Take, {1, 2})
    };

    auto c = analyze(p);
    EXPECT_TRUE(static_cast<bool>(c));

    EXPECT_EQ(2, c.value().inputs.size());
    EXPECT_EQ(0, c.value().inputs[0]);
    EXPECT_EQ(1, c.value().inputs[1]);

    EXPECT_TRUE(static_cast<bool>(c.value().integer_variables.find(1)->second.min));
    EXPECT_EQ(0, c.value().integer_variables.find(1)->second.min.value());
}

TEST(GenerateExamplesTest, SimpleGenerationTest) {
    auto p = {
            Statement(0, Function::ReadList, {}),
            Statement(1, Function::ReadInt, {}),
            Statement(2, Function::Map, {OneArgumentLambda::Plus1, 0}),
            Statement(3, Function::Take, {1, 2})
    };


    for (auto i = 0; i < 100; i++) {
        auto examples = generate_examples(p);

        EXPECT_TRUE(static_cast<bool>(examples));
        EXPECT_TRUE(examples.value().size() > 0);
        for (auto &example: examples.value()) {
            EXPECT_EQ(eval(p, example.input), example.output);
        }
    }
}