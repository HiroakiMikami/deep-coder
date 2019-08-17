import dataclasses
from enum import Enum
from typing import List, Tuple, TypeVar
import importlib
generate_io_samples = importlib.import_module("DeepCoder_Utils.generate_io_samples")
Function = generate_io_samples.Function

class Type(Enum):
    Int = 1
    IntList = 2

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
    body: List[Tuple[Variable, Expression]]

def to_string(program: Program) -> str:
    """
    Return the source code of the program

    Parameters
    ----------
    program : Program
        The program that will be converted to the string

    Returns
    -------
    code : string
        The source code of `program`
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

    for input in program.inputs:
        code += "{} <- {}\n".format(id_to_name(input.id), "int" if input.t == Type.Int else "[int]")
    for v, exp in program.body:
        code += "{} <- {} {}\n".format(id_to_name(v.id), exp.function.src, " ".join(map(lambda x: id_to_name(x.id), exp.arguments)))

    return code
