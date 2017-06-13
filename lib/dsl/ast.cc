#include "dsl/ast.h"

using namespace dsl;

std::vector<Function> dsl::all_functions = {Function::Head, Function::Last, Function::Take, Function::Drop,
                                            Function::Access, Function::Minimum, Function::Maximum, Function::Reverse,
                                            Function::Sort, Function::Sum, Function::Map, Function::Filter,
                                            Function::Count, Function::ZipWith, Function::Scanl1, Function::ReadInt,
                                            Function::ReadList};

std::vector<PredicateLambda> dsl::all_predicate_lambdas = {PredicateLambda::IsPositive,
                                                           PredicateLambda::IsNegative,
                                                           PredicateLambda::IsEven,
                                                           PredicateLambda::IsOdd};

std::vector<TwoArgumentsLambda> dsl::all_two_arguments_lambdas = {TwoArgumentsLambda::Plus,
                                                                  TwoArgumentsLambda::Minus,
                                                                  TwoArgumentsLambda::Multiply,
                                                                  TwoArgumentsLambda::Min,
                                                                  TwoArgumentsLambda::Max};

std::vector<OneArgumentLambda> dsl::all_one_argument_lambdas = {OneArgumentLambda::Plus1,
                                                                OneArgumentLambda::Minus1,
                                                                OneArgumentLambda::MultiplyMinus1,
                                                                OneArgumentLambda::Multiply2,
                                                                OneArgumentLambda::Multiply3,
                                                                OneArgumentLambda::Multiply4,
                                                                OneArgumentLambda::Divide2,
                                                                OneArgumentLambda::Divide3,
                                                                OneArgumentLambda::Divide4,
                                                                OneArgumentLambda::Pow2};

Argument::Argument(Variable variable) : m_argument(variable) {}
Argument::Argument(TwoArgumentsLambda lambda) : m_argument(static_cast<uint32_t>(lambda)) {}
Argument::Argument(PredicateLambda lambda) : m_argument(static_cast<uint32_t>(lambda)) {}
Argument::Argument(OneArgumentLambda lambda) : m_argument(static_cast<uint32_t>(lambda)) {}

std::experimental::optional<PredicateLambda> Argument::predicate() const {
    if (this->m_argument & 0x40000000) {
        return static_cast<PredicateLambda>(this->m_argument);
    } else {
        return {};
    }
}
std::experimental::optional<TwoArgumentsLambda> Argument::two_arguments_lambda() const {
    if (this->m_argument & 0x20000000) {
        return static_cast<TwoArgumentsLambda >(this->m_argument);
    } else {
        return {};
    }
}
std::experimental::optional<OneArgumentLambda> Argument::one_argument_lambda() const {
    if (this->m_argument & 0x10000000) {
        return static_cast<OneArgumentLambda>(this->m_argument);
    } else {
        return {};
    }
}
std::experimental::optional<Variable> Argument::variable() const {
    if (this->m_argument & 0x70000000) {
        return {};
    } else {
        return static_cast<uint16_t>(this->m_argument & 0xffff);
    }
}

Statement::Statement(Variable variable, Function function, const std::vector<Argument> &arguments)
        : variable(variable), function(function), arguments(arguments) {}