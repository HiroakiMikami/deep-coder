import dataclasses
import copy
import pickle
import os
import contextlib
import numpy as np
from typing import List, Tuple, Union, Dict, Callable
from .deepcoder_utils import generate_io_samples
from .dsl import Function, Program, to_string, clone, Type, to_function
from .source_code_simplifier import normalize
from .source_code_generator import source_code

Primitive = Union[int, List[int]]
Example = Tuple[List[Primitive], Primitive]

@dataclasses.dataclass
class Entry:
    """
    The entry of this dataset

    Attributes
    ----------
    source_code : str
        The source code of the program
    examples : list of Example
        The input/output examples for the source code
    attributes : dict from str to bool
        The binary attributes of the source code.
        The strings of keys represent the functions, and the value
        represents whether the program contains the function or not.
    """
    source_code: str
    examples: List[Example]
    attributes: Dict[str, bool]

@dataclasses.dataclass
class Dataset:
    """
    The dataset

    Attributes
    ----------
    entries : list of Entry
        The entries of this dataset.
    """
    entries: List[Entry]

@dataclasses.dataclass
class DatasetSpec:
    """
    The specification of the dataset

    Attribute
    ---------
    value_range : int
    max_list_length : int
    num_examples : int
    min_program_length : int
    max_program_length : int
    """
    value_range: int
    max_list_length: int
    num_examples: int
    min_program_length: int
    max_program_length: int

@dataclasses.dataclass
class EquivalenceCheckingSpec:
    """
    The specification used to check equivalence of programs

    Attribute
    ---
    ratio_of_examples : float
    num_of_examples : int
    rng : np.random.RandomState or None
    """
    ratio_of_examples: float
    num_of_examples: int
    rng: Union[np.random.RandomState, None]

SimplifyFunction = Callable[[Program], Program]

@dataclasses.dataclass
class ProgressCallback:
    """
    The callback functions to receive the progress of data generation

    Attribute
    ---
    on_generate_program : Callable[[Program], None]
    on_finish_enumeration : Callabke[[int], None]
    on_dump_dataset : Callable[[int], None]
    """
    on_generate_program: Callable[[Program], None]
    on_finish_enumeration: Callable[[int], None]
    on_dump_dataset : Callable[[int], None]

def generate_dataset(functions: List[generate_io_samples.Function], spec: DatasetSpec,
                     equivalence_spec: EquivalenceCheckingSpec,
                     destinationDir: str, simplify: Union[None, SimplifyFunction]=None,
                     callback: Union[None, ProgressCallback] = None):
    """
    Generate dataset to the file

    Parameters
    ----------
    functions : list of generate_io_samples.Function
        The set of functions that can be used in the dataset
    spec : DatasetSpec
        The specification of generated dataset
    equivalence_spec: EquivalenceCheckingSpec
        The specification used to check equivalence of programs
    destinationDir : str
        The destination of the dataset file
    simplify : function or None
        The function to simplify the source code

    Notes
    -----
    Currently this function generates and prunes source code in memory.
    It might be a problem if the program size is large.
    """

    @dataclasses.dataclass
    class IntermidiateEntry:
        source_code: str
        program: generate_io_samples.Program
        examples: List[Example]
        attributes: Dict[str, bool]

    def simplify_and_normalize(program: Program) -> Program:
        while True:
            p_old = to_string(program)
            if simplify is not None:
                program = simplify(program)

            if to_string(program) == p_old:
                break
        program = normalize(program)
        return program

    def get_signature(program: Program):
        input = []
        for i in program.inputs:
            input.append(i.t)
        output = program.body[-1][1].function.signature[-1] if len(program.body) > 0 else None
        return (input, output)
    def signature_to_string(signature):
        return "{}".format(signature)

    functions_dsl = [to_function(f) for f in functions]
    invalid_program = set()
    entries = dict() # signature -> dict(str -> IntermidiateEntry)

    # Enumerate source code
    n_programs = 0
    for program in source_code(functions_dsl, spec.min_program_length, spec.max_program_length):
        program = simplify_and_normalize(program) # Simplify the program
        if not (spec.min_program_length <= len(program.body) <= spec.max_program_length):
            # If the length of simplified program is out of range, discard the program
            continue

        signature = signature_to_string(get_signature(program))
        if not signature in entries:
            entries[signature] = dict()

        code = to_string(program)[:-1] # last newline should be removed to compile source code

        if code in entries[signature]:
            # the program is already added to the dataset
            continue

        # Compile the source code
        if code in invalid_program:
            continue
        with contextlib.redirect_stdout(None): # ignore stdout
            p = generate_io_samples.compile(code, V=spec.value_range, L=spec.max_list_length)
        if p is None:
            # Compilation is failed
            invalid_program.add(code)
            continue

        try:
            # Generate IO examples
            with contextlib.redirect_stdout(None): # ignore stdout
                examples = generate_io_samples.generate_IO_examples(p, N=spec.num_examples, L=spec.max_list_length, V=spec.value_range)
        except ValueError:
            continue

        # Generate binary attributes
        fs = set()
        for _, exp in program.body:
            fs.add(exp.function.name)
        attributes = dict()
        for f in functions_dsl:
            attributes[f.name] = f.name in fs

        if callback is not None:
            callback.on_generate_program(program)
        n_programs += 1
        entries[signature][code] = IntermidiateEntry(code, p, examples, attributes)

    if callback is not None:
        callback.on_finish_enumeration(n_programs)

    # Prune entries
    rng = equivalence_spec.rng if equivalence_spec.rng is not None else np.random
    for signature, ientries in entries.items():
        examples: List[List[Primitive]] = list()
        # Extract examples for checking equivalence
        num = max(
            1,
            equivalence_spec.num_of_examples,
            int(len(ientries) * spec.num_examples * equivalence_spec.ratio_of_examples))
        num = min(num, len(ientries) * spec.num_examples)

        from_all_entries = num // len(ientries)
        from_partial_entries = num % len(ientries)

        # Extract examples from all entries
        not_used = dict() # str -> [int]
        for entry in ientries.values():
            indexes = set(rng.choice(list(range(spec.num_examples)), from_all_entries, replace=False))
            for index in indexes:
                examples.append(entry.examples[index][0])
            not_used[entry.source_code] = [i for i in range(spec.num_examples) if not (i in indexes) ]
        # Extract examples from partial entries
        if from_partial_entries != 0:
            for entry in rng.choice(list(ientries.values()), from_partial_entries, replace=False):
                index = rng.choice(not_used[entry.source_code])
                examples.append(entry.examples[index][0])
        
        # Execute programs
        es = dict() # Tuple[Primitive] -> IntermidiateEntry
        for entry in ientries.values():
            result = []
            for example in examples:
                output = entry.program.fun(example)
                if entry.program.out == int:
                    result.append(output)
                else:
                    result.append(tuple(output))
            result = tuple(result)
            if not result in es:
                es[result] = entry
            else:
                # If there is a equivalent program, prune the longer program
                l1 = len(es[result].source_code.split("\n"))
                l2 = len(entry.source_code.split("\n"))
                if l1 > l2:
                    es[result] = entry

        # Create dataset instance
        d = Dataset([])
        for entry in es.values():
            d.entries.append(Entry(
                entry.source_code, entry.examples, entry.attributes
        ))
        if callback is not None:
            callback.on_dump_dataset(len(ientries))

        # Dump the dataset to the file
        with open(os.path.join(destinationDir, "{}.pickle".format(signature)), "wb") as f:
            pickle.dump(d, f)
