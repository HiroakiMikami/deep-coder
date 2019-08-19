import dataclasses
import copy
import pickle
from typing import List, Tuple, Union, Dict, Callable, BinaryIO
from .deepcoder_utils import generate_io_samples
from .dsl import Function, Program, to_string, clone
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
    attributes: Dict[Function, bool]

@dataclasses.dataclass
class Dataset:
    """
    The dataset

    Attributes
    ----------
    entries : dict from string to list of Entry
        The entries of this dataset.
        The strings of keys represent the type signature of the program.
    """
    entries: Dict[str, List[Entry]]

@dataclasses.dataclass
class DatasetSpec:
    """
    The specification of the dataset

    Attribute
    ---------
    functions : list of Function
        The set of functions that can be used in this dataset
    value_range : int
    max_list_length : int
    num_examples : int
    min_program_length : int
    max_program_length : int
    """
    functions: List[Function]
    value_range: int
    max_list_length: int
    num_examples: int
    min_program_length: int
    max_program_length: int

SimplifyFunction = Callable[[Program], Program]

def generate_dataset(spec: DatasetSpec, destination: BinaryIO, simplify: Union[None, SimplifyFunction]=None):
    """
    Generate dataset to the file

    Parameters
    ----------
    spec : DatasetSpec
        The specification of generated dataset
    destination : file object
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
        attributes: Dict[Function, bool]

    def simplify_and_normalize(program: Program) -> Program:
        while True:
            p_old = to_string(program)
            if simplify is not None:
                program = simplify(program)

            if to_string(program) == p_old:
                break
        program = normalize(program)
        return program

    def signature_to_string(signature):
        return "{}".format(signature)

    def is_identical(entry1: Entry, entry2: Entry) -> bool:
        """
        Check whether the program of entry1 and entry2 are identical or not
        """
        for input, output in entry1.examples:
            if output != entry2.program.fun(input):
                return False
        for input, output in entry2.examples:
            if output != entry1.program.fun(input):
                return False
        return True

    dataset = {} # Signature -> (string -> IntermidiateEntry)
    invalid_program = set()
    # Enumerate source code
    for program in source_code(spec.functions, spec.min_program_length, spec.max_program_length):
        program = simplify_and_normalize(program) # Simplify the program
        if not (spec.min_program_length <= len(program.body) <= spec.max_program_length):
            # If the length of simplified program is out of range, discard the program
            continue
        s = to_string(program)[:-1] # last newline should be removed to compile source code

        # Compile the source code
        if s in invalid_program:
            continue
        p = generate_io_samples.compile(s, V=spec.value_range, L=spec.max_list_length)
        if p is None:
            # Compilation is failed
            invalid_program.add(s)
            continue

        signature = signature_to_string((tuple(p.ins), p.out))
        if not signature in dataset:
            dataset[signature] = dict()

        if s in dataset[signature]:
            # the program is already added to the dataset
            continue
        
        try:
            # Generate IO examples
            examples = generate_io_samples.generate_IO_examples(p, N=spec.num_examples, L=spec.max_list_length, V=spec.value_range)
        except ValueError:
            continue

        # Generate binary attributes
        used_functions = set()
        for _, exp in program.body:
            used_functions.add(exp.function.src)
        attributes = dict()
        for f in spec.functions:
            attributes[f.src] = f.src in used_functions
        entry = IntermidiateEntry(s, p, examples, attributes)

        # Prune entries
        removed = set()
        add_s = True
        for s2, entry2 in dataset[signature].items():
            if is_identical(entry, entry2):
                if len(entry.source_code.split("\n")) >= len(entry2.source_code.split("\n")):
                    # This entry should be pruned
                    add_s = False
                    break
                else:
                    # entry2 should be removed
                    removed.add(s2)
        if add_s:
            dataset[signature][s] = entry
        for r in removed:
            del dataset[signature][r]
    
    # Create dataset instance
    d = Dataset(dict())
    for signature, entries in dataset.items():
        d.entries[signature] = []
        for entry in entries.values():
            d.entries[signature].append(Entry(
                entry.source_code,
                entry.examples,
                entry.attributes
            ))

    # Dump the dataset to the file
    pickle.dump(d, destination)
