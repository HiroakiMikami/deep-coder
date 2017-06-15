#include <iostream>
#include <gtest/gtest.h>
#include "dataset-generator.h"

using namespace std;

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
    EXPECT_FALSE(x);
}

TEST(GenerateIntegerTest, SignTest) {
    IntegerConstraint c;
    c.min = -10;
    c.max = 20;

    c.sign = 1;
    for (auto i = 0; i < 100; i++) {
        auto x = generate_integer(c);
        EXPECT_TRUE(x >= 1);
        EXPECT_TRUE(x <= 20);
    }

    c.sign = -1;
    for (auto i = 0; i < 100; i++) {
        auto x = generate_integer(c);
        EXPECT_TRUE(x >= -10);
        EXPECT_TRUE(x <= -1);
    }

    c.sign = 0;
    for (auto i = 0; i < 100; i++) {
        auto x = generate_integer(c);
        EXPECT_EQ(0, x);
    }
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
        EXPECT_TRUE(x);
        EXPECT_TRUE(x.value().size() >= 10);
        EXPECT_TRUE(x.value().size() <= 20);
    }

    c.max_length = 5;
    auto x = generate_list(c);
    EXPECT_FALSE(x);
}

TEST(GenerateListTest, IntegerConstraintTest) {
    ListConstraint c;
    c.min_length = 1;

    c.min = 5;
    c.max = 10;
    c.is_even = {true};

    for (auto i = 0; i < 100; i++) {
        auto x = generate_list(c);
        EXPECT_TRUE(x);
        for (const auto &i: x.value()) {
            EXPECT_TRUE(i >= 5);
            EXPECT_TRUE(i <= 10);
            EXPECT_TRUE((i % 2) == 0);
        }
    }

    c.min = -1;
    c.max = 0;
    c.sign = {1};

    auto x = generate_list(c);
    EXPECT_FALSE(x);
}