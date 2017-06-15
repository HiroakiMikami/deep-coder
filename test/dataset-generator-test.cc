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