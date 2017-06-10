#include <string>
#include <algorithm>
#include "dsl/utils.h"

using namespace std;

namespace dsl {
    string stringify(const Function &func) {
        switch (func) {
            case Function::Head:
                return "head";
            case Function::Last:
                return "last";
            case Function::Take:
                return "take";
            case Function::Drop:
                return "drop";
            case Function::Access:
                return "access";
            case Function::Minimum:
                return "minimum";
            case Function::Maximum:
                return "maximum";
            case Function::Reverse:
                return "reverse";
            case Function::Sort:
                return "sort";
            case Function::Sum:
                return "sum";
            case Function::Map:
                return "map";
            case Function::Filter:
                return "filter";
            case Function::Count:
                return "count";
            case Function::ZipWith:
                return "zip_with";
            case Function::Scanl1:
                return "scanl1";
            case Function::ReadInt:
                return "read_int";
            case Function::ReadList:
                return "read_list";
        }
    }

    std::string stringify(const OneArgumentLambda &lambda) {
        switch (lambda) {
            case OneArgumentLambda::Plus1:
                return "+1";
            case OneArgumentLambda::Minus1:
                return "-1";
            case OneArgumentLambda::Multiply2:
                return "*2";
            case OneArgumentLambda::Multiply3:
                return "*3";
            case OneArgumentLambda::Multiply4:
                return "*4";
            case OneArgumentLambda::MultiplyMinus1:
                return "*(-1)";
            case OneArgumentLambda::Divide2:
                return "/2";
            case OneArgumentLambda::Divide3:
                return "/3";
            case OneArgumentLambda::Divide4:
                return "/4";
            case OneArgumentLambda::Pow2:
                return "**2";
        }
    }

    std::string stringify(const TwoArgumentsLambda &lambda) {
        switch (lambda) {
            case TwoArgumentsLambda::Plus:
                return "+";
            case TwoArgumentsLambda::Minus:
                return "-";
            case TwoArgumentsLambda::Multiply:
                return "*";
            case TwoArgumentsLambda::Min:
                return "MIN";
            case TwoArgumentsLambda::Max:
                return "MAX";
        }
    }

    std::string stringify(const PredicateLambda &pred) {
        switch (pred) {
            case PredicateLambda::IsPositive:
                return ">0";
            case PredicateLambda::IsNegative:
                return "<0";
            case PredicateLambda::IsOdd:
                return "%2 == 1";
            case PredicateLambda::IsEven:
                return "%2 == 0";
        }
    }

    std::string stringify(const Variable &var) {
        std::string str;

        Variable x = var;

        while (x >= 26) {
            str += (x % 26) + 'a';
            x /= 26;
        }

        if (var >= 26) {
            x -= 1;
        }

        str += x + 'a';

        reverse(str.begin(), str.end());

        return str;
    }

    std::ostream &operator<<(std::ostream &stream, const Argument &argument) {
        if (argument.one_argument_lambda()) {
            stream << stringify(argument.one_argument_lambda().value());
        } else if (argument.two_arguments_lambda()) {
            stream << stringify(argument.two_arguments_lambda().value());
        } else if (argument.predicate()) {
            stream << stringify(argument.predicate().value());
        } else if (argument.variable()) {
            stream << stringify(argument.variable().value());
        }

        return stream;
    }

    std::ostream &operator<<(std::ostream &stream, const Statement &statement) {
        stream << stringify(statement.variable) << " <- " << stringify(statement.function);
        for (const auto &arg: statement.arguments) {
            stream << " " << arg;
        }

        return stream;
    }

    std::ostream &operator<<(std::ostream &stream, const Program &program) {
        stream << "---\n";
        for (const auto &statement: program) {
            stream << statement << "\n";
        }
        stream << "---\n";

        return stream;
    }

    std::ostream &operator<<(std::ostream &stream, const Value &value) {
        if (value.integer()) {
            stream << value.integer().value();
        } else if (value.list()) {
            stream << "[";
            auto l = value.list().value();

            for (auto i = 0; i < l.size(); i++) {
                if (i != 0) {
                    stream << ",";
                }
                stream << l[i];
            }

            stream << "]";
        } else {
            stream << "NULL";
        }

        return stream;
    }
    std::ostream &operator<<(std::ostream &stream, const Input &input) {
        stream << "---\n";

        for (const auto& i: input) {
            stream << i << "\n";
        }

        stream << "---\n";

        return stream;
    }

}