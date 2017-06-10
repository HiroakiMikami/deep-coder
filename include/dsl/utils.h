#include <iostream>
#include "dsl/ast.h"

#pragma once

namespace dsl {
    std::ostream &operator<<(std::ostream &stream, const Argument &argument);
    std::ostream &operator<<(std::ostream &stream, const Statement &statement);
    std::ostream &operator<<(std::ostream &stream, const Program &program);
}