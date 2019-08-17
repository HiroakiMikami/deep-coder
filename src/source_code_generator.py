import dataclasses
from typing import List, Set, Union
import copy
from .dsl import Function, Type, Variable, Expression, Program, clone

class IdGenerator:
    """
    The class to generate a unique id

    Attributes
    ----------
    _n : int
        The next id
    """
    def __init__(self):
        self._n = 0
    def generate(self):
        """
        Return a unique id

        Returns
        -------
        id : int
            A generated unique id
        """
        id = self._n
        self._n += 1
        return id

@dataclasses.dataclass
class ArgumentWithState:
    arguments: List[Variable]
    generator: IdGenerator
    variables: Set[Variable]
    new_variables: List[Variable]

def arguments(id_generator: IdGenerator, variables: Set[Variable], signature):
    """
    Enumerate all arguments that match the signature

    Parameters
    ----------
    id_generator : IdGenerator
        The generator used to create new variables
    variables : set of Variable
        The set of variables that are currently defined
    signature : list of ([int] or int)
        The signature of the arguments.
        `int` represents Type.Int and `[int]` represents Type.IntList.

    Yields
    ------
    ArgumentWithState
        The argument list and the state that will be used to continue enumeration
    """

    # Perform DFS to enumerate arguments
    s = list([ArgumentWithState([], id_generator, variables, set())]) # Start from an empty list
    while len(s) != 0:
        elem = s.pop()

        if len(elem.arguments) == len(signature):
            yield elem
        else:
            t_arg = Type.Int if signature[len(elem.arguments)] == int else Type.IntList # The type of the argument
            candidates = [v for v in elem.variables if v.t == t_arg] # Existing variables which type is t_arg
            
            for v in candidates:
                # Use existing var
                s.append(ArgumentWithState([*(elem.arguments), v], elem.generator, elem.variables, elem.new_variables))
            # Create new var
            generator_new = copy.deepcopy(elem.generator)
            v_new = Variable(generator_new.generate(), t_arg)
            arg_new = [*(elem.arguments), v_new]
            vars_new = set([*(elem.variables), v_new])
            s.append(ArgumentWithState(arg_new, generator_new, vars_new, [*(elem.new_variables), v_new]))

def source_code(functions: List[Function], min_length: int, max_length: int):
    """
    Enumerate all source code which length is in [min_length:max_length]

    Parameters
    ----------
    functions : list of Function
        All functions that can be used in source code
    min_length : int
        The minimum length of programs
    max_length : int
        The maximum length of programs

    Yields
    ------
    Program
        The program which length is in [min_length:max_length]
    """
    assert(min_length <= max_length)

    def type_to_enum(t):
        return Type.Int if t == int else Type.IntList

    # Perform DFS to enumerate source code
    s = [(Program([], []), IdGenerator())] # Start from a program with no expressions
    while len(s) != 0:
        p, g = s.pop()

        if min_length <= len(p.body) <= max_length:
            yield p
        if len(p.body) >= max_length:
            continue

        # Create a set of variables
        vars = set(p.inputs + list(map(lambda x: x[0], p.body)))
        
        # Enumerate functions
        for func in functions:
            # Enumerate arguments
            for a in arguments(g, vars, func.sig[:-1]):
                p_new = copy.deepcopy(p)
                for v in a.new_variables:
                    p_new.inputs.append(v)
                generator = copy.deepcopy(a.generator)
                p_new.body.append((Variable(generator.generate(), type_to_enum(func.sig[-1])), Expression(func, a.arguments)))
                s.append((p_new, generator))
