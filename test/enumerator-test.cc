#include <gtest/gtest.h>
#include <algorithm>
#include "enumerator.h"

#include <iostream>
#include "dsl/utils.h"
using namespace std;
using namespace dsl;

TEST(EnumerateTest, EnumerateFunctionTest) {
    Restriction r = {
            1, 2,
            { Function::ReadInt, Function::ReadList },
            {}, {}, {}
    };

    vector<Program> ps;
    auto process = [&ps](const Program &p, const int &x) {
        ps.push_back(p);

        return true;
    };
    enumerate(r, [](const Program &p, const int &x) { return p.size(); }, process, 0);

    EXPECT_EQ(6, ps.size());
    EXPECT_TRUE(find_if(ps.begin(), ps.end(), [](const Program& p) -> bool {
        return p[0].function == Function::ReadInt;
    }) != ps.end());
}

TEST(EnumerateTest, EnumerateArgumentsTest) {
    Restriction r = {
            1, 2,
            { Function::ReadList, Function::Head },
            {}, {}, {}
    };

    vector<Program> ps;
    auto process = [&ps](const Program &p, const int &x) {
        ps.push_back(p);

        return true;
    };
    enumerate(r, [](const Program &p, const int &x) { return p.size(); }, process, 0);

    EXPECT_EQ(3, ps.size());
    EXPECT_TRUE(find_if(ps.begin(), ps.end(), [](const Program& p) -> bool {
        return p.size() == 2 && p[1].function == Function::Head && p[1].arguments[0].variable().value() == 0;
    }) != ps.end());
}

TEST(EnumerateTest, EnumerateLambdaTest) {
    Restriction r = {
            1, 2,
            { Function::ReadList, Function::Count },
            { PredicateLambda::IsNegative, PredicateLambda::IsPositive}, {}, {}
    };

    vector<Program> ps;
    auto process = [&ps](const Program &p, const int &x) {
        ps.push_back(p);

        return true;
    };
    enumerate(r, [](const Program &p, const int &x) { return p.size(); }, process, 0);

    EXPECT_EQ(4, ps.size());
    EXPECT_TRUE(find_if(ps.begin(), ps.end(), [](const Program& p) -> bool {
        return p.size() == 2 &&
                p[1].function == Function::Count && p[1].arguments[0].predicate().value() == PredicateLambda::IsNegative;
    }) != ps.end());
    EXPECT_TRUE(find_if(ps.begin(), ps.end(), [](const Program& p) -> bool {
        return p.size() == 2 &&
               p[1].function == Function::Count && p[1].arguments[0].predicate().value() == PredicateLambda::IsPositive;
    }) != ps.end());
}

TEST(EnumerateTest, MinLengthTest) {
    Restriction r = {
            2, 2,
            { Function::ReadInt, Function::ReadList },
            {}, {}, {}
    };

    vector<Program> ps;
    auto process = [&ps](const Program &p, const int &x) {
        ps.push_back(p);

        return true;
    };
    enumerate(r, [](const Program &p, const int &x) { return p.size(); }, process, 0);

    EXPECT_EQ(4, ps.size());
}

TEST(EnumerateTest, InformationTest) {
    Restriction r = {
            2, 2,
            { Function::ReadInt, Function::ReadList },
            {}, {}, {}
    };

    vector<Program> ps;
    auto process = [&ps](const Program &p, const int &x) {
        EXPECT_EQ(p.size(), x + 1);
        ps.push_back(p);

        return true;
    };
    enumerate(r, [](const Program &p, const int &x) { return x + 1; }, process, 0);

    EXPECT_EQ(4, ps.size());
}

TEST(EnumerateTest, NoDuplicateProgramTest) {
    Restriction r1 = {
            1, 1,
            { Function::ReadInt, Function::ReadList },
            {}, {}, {}
    };
    Restriction r2 = {
            2, 2,
            { Function::Head },
            {}, {}, {}
    };

    std::vector<Program> ps;

    enumerate(r1, [](const Program &p, const int &x) { return x; }, [&](const Program &p, const int &x) {
        enumerate(r2, [](const Program &p, const int &x) { return x; }, [&](const Program &p, const int &x) {
            ps.push_back(p);
            return true;
        }, p, 0);
        return true;
    }, 0);

    // ReadList -> Head
    EXPECT_EQ(1, ps.size());
}

TEST(EnumerateTest, NoDuplicateProgramTest2) {
    Restriction r1 = {
            2, 2,
            { Function::ReadList },
            {}, {}, {}
    };
    Restriction r2 = {
            3, 3,
            { Function::ZipWith },
            {}, {}, {TwoArgumentsLambda::Plus}
    };

    std::vector<Program> ps;

    enumerate(r1, [](const Program &p, const int &x) { return x; }, [&](const Program &p, const int &x) {
        enumerate(r2, [](const Program &p, const int &x) { return x; }, [&](const Program &p, const int &x) {
            ps.push_back(p);
            return true;
        }, p, 0);
        return true;
    }, 0);

    // Case 1
    //  a <- ReadList
    //  b <- ReadList
    //  c <-ZipWith + a b
    // Case 2
    //  a <- ReadList
    //  b <- ReadList
    //  c <-ZipWith + b a
    // Case 3
    //  a <- ReadList
    //  b <- ReadList
    //  c <-ZipWith + a a
    // Case 4
    //  a <- ReadList
    //  b <- ReadList
    //  c <-ZipWith + b b
    EXPECT_EQ(4, ps.size());
}
TEST(EnumerateTest, BreakTest) {
    Restriction r1 = {
            2, 2,
            { Function::ReadList, Function::ReadInt },
            {}, {}, {}
    };

    std::vector<Program> ps;

    auto num = 0;
    enumerate(r1, [](const Program &p, const int &x) { return x; }, [&](const Program &p, const int &x) {
        num += 1;
        return false;
    }, 0);

    EXPECT_EQ(1, num);
}
