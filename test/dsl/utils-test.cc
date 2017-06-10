#include <gtest/gtest.h>
#include <sstream>
#include "dsl/utils.h"

using namespace dsl;
using namespace std;

TEST(OutputTest, ArgumentOutputTest) {
    ostringstream oss;
    oss << Argument(OneArgumentLambda::Pow2);

    EXPECT_EQ("**2", oss.str());
}

TEST(OutputTest, StatementOutputTest) {
    ostringstream oss;
    oss << Statement(0, Function::Head, vector<Argument>({Argument(27)}));

    EXPECT_EQ("a <- head ab", oss.str());
}

TEST(OutputTest, ProgramOutputTest) {
    ostringstream oss;
    oss << Program({Statement(0, Function::Head, vector<Argument>({Argument(27)}))});

    EXPECT_EQ("---\na <- head ab\n---\n", oss.str());
}

TEST(OutputTest, ValueOutputTest) {
    ostringstream oss;
    oss << Value({0, 1, 2});

    EXPECT_EQ("[0,1,2]", oss.str());
}

TEST(OutputTest, InputOutputTest) {
    ostringstream oss;
    oss << Input({Value({0, 1, 2}), Value(0)});

    EXPECT_EQ("---\n[0,1,2]\n0\n---\n", oss.str());
}