#include "dsl/ast.h"

using namespace dsl;

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