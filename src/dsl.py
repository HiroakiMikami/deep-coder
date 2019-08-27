import dataclasses
from enum import Enum
from typing import List
import copy
from src.deepcoder_utils import generate_io_samples


class Type(Enum):
    Int = 1
    IntList = 2


@dataclasses.dataclass
class Signature:
    """
    The function signature of DSL programs

    Attributes
    ----------
    input_types: List[Type]
    output_type: Type
    """
    input_types: List[Type]
    output_type: Type

    def __eq__(self, rhs):
        return self.input_types == rhs.input_types and self.output_type == rhs.output_type

    def __hash__(self):
        return hash((tuple(self.input_types), self.output_type))


@dataclasses.dataclass
class Function:
    """
    The function of DSL programs

    Attributes
    ----------
    name : str
        The name of this function
    signature : Signature
        The signature of this function
    """
    name: str
    signature: Signature

    def __eq__(self, rhs):
        return self.name == rhs.name and self.signature == rhs.signature

    def __hash__(self):
        return hash((self.name, (tuple(self.signature.input_types), self.signature.output_type)))


@dataclasses.dataclass
class Variable:
    """
    The variable of DSL programs

    Attributes
    ----------
    id : int
        The identifier of this variable
    t : Type
        The type of this variable
    """

    id: int
    t: Type

    def __eq__(self, rhs):
        return self.id == rhs.id and self.t == rhs.t

    def __hash__(self):
        return hash((self.id, self.t))


@dataclasses.dataclass
class Expression:
    """
    The expression of DSL programs

    Attributes
    ----------
    function : Function
        The function that this expression calls
    arguments: list of Variable
        The arguments of the function call
    """

    function: Function
    arguments: List[Variable]


@dataclasses.dataclass
class Statement:
    """
    The statement of DSL

    Attributes
    ----------
    variable : Variable
        The variable defined in this statement
    expression : Expression
    """

    variable: Variable
    expression: Expression


@dataclasses.dataclass
class Program:
    """
    The program of DSL

    Attributes
    ----------
    inputs : list of Variable
        The input variables of this program
    body : list of (Variable, Expression)
        The body of this program.
        The interpreter will execute an expression and store the result
        to the variablefor each element of the list.
    """

    inputs: List[Variable]
    body: List[Statement]

    def to_string(self) -> str:
        """
        Return the source code of the program

        Returns
        -------
        code : string
            The source code of this program
        """

        code = ""

        def id_to_name(id: int) -> str:
            name = ""
            while True:
                x = id % 26
                id //= 26
                name += chr(x + ord('a'))
                if id == 0:
                    break
            return name

        for input in self.inputs:
            code += "{} <- {}\n".format(id_to_name(input.id),
                                        "int" if input.t == Type.Int else "[int]")
        for statement in self.body:
            code += "{} <- {} {}\n".format(id_to_name(statement.variable.id), statement.expression.function.name, " ".join(
                map(lambda x: id_to_name(x.id), statement.expression.arguments)))

        return code

    def clone(self):
        """
        Return the copy of the program
        The self and the return value will not share any objects.
        Thus we can freely modify the return value.

        Returns
        -------
        cloned_program : Program
            The program that is same as this program
        """

        inputs = []
        body = []
        for input in self.inputs:
            inputs.append(copy.deepcopy(input))
        for statement in self.body:
            args = []
            for arg in statement.expression.arguments:
                args.append(copy.deepcopy(arg))
            body.append(Statement(copy.deepcopy(statement.variable),
                                  Expression(statement.expression.function, args)))

        return Program(inputs, body)


def to_function(f: generate_io_samples.Function) -> Function:
    """
    Convert from generate_io_samples.Function to dsl.Function

    Parameters
    ----------
    f : generate_io_samples.Function
        The function that will be converted

    Returns
    -------
    Function
        The converted Function instance
    """
    intype = []
    for s in f.sig[:-1]:
        intype.append(Type.Int if s == int else Type.IntList)
    outtype = Type.Int if f.sig[-1] == int else Type.IntList
    return Function(f.src, Signature(intype, outtype))
