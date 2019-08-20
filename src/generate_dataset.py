import dataclasses
import copy
import pickle
import os
import multiprocessing as mp
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

SimplifyFunction = Callable[[Program], Program]

def worker(functions: List[Function], spec: DatasetSpec, destinationDir: str, signature: str, queue: mp.SimpleQueue):
    @dataclasses.dataclass
    class IntermidiateEntry:
        source_code: str
        program: generate_io_samples.Program
        examples: List[Example]
        attributes: Dict[str, bool]

    def is_identical(entry1: IntermidiateEntry, entry2: IntermidiateEntry) -> bool:
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

    dataset = dict()
    invalid_program = set()
    while True:
        # Get a next program
        program = queue.get()

        if program is None:
            # Finish this worker
            break
        code = to_string(program)[:-1] # last newline should be removed to compile source code
        fs = set()
        for _, exp in program.body:
            fs.add(exp.function.name)

        if code in dataset:
            # the program is already added to the dataset
            continue

        # Compile the source code
        if code in invalid_program:
            continue
        p = generate_io_samples.compile(code, V=spec.value_range, L=spec.max_list_length)
        if p is None:
            # Compilation is failed
            invalid_program.add(code)
            continue

        try:
            # Generate IO examples
            examples = generate_io_samples.generate_IO_examples(p, N=spec.num_examples, L=spec.max_list_length, V=spec.value_range)
        except ValueError:
            continue

        # Generate binary attributes
        attributes = dict()
        for f in functions:
            attributes[f.name] = f.name in fs

        entry = IntermidiateEntry(code, p, examples, attributes)

        # Prune entries
        removed = set()
        add_s = True
        for s2, entry2 in dataset.items():
            if is_identical(entry, entry2):
                if len(entry.source_code.split("\n")) >= len(entry2.source_code.split("\n")):
                    # This entry should be pruned
                    add_s = False
                else:
                    # entry2 should be removed
                    removed.add(s2)
                break
        if add_s:
            dataset[code] = entry
        for r in removed:
            del dataset[r]

    # Create dataset instance
    d = Dataset([])
    for entry in dataset.values():
        d.entries.append(Entry(
            entry.source_code, entry.examples, entry.attributes
    ))

    # Dump the dataset to the file
    with open(os.path.join(destinationDir, "{}.pickle".format(signature)), "wb") as f:
        pickle.dump(d, f)


def generate_dataset(functions: List[generate_io_samples.Function], spec: DatasetSpec, destinationDir: str, simplify: Union[None, SimplifyFunction]=None):
    """
    Generate dataset to the file

    Parameters
    ----------
    functions : list of generate_io_samples.Function
        The set of functions that can be used in the dataset
    spec : DatasetSpec
        The specification of generated dataset
    destinationDir : str
        The destination of the dataset file
    simplify : function or None
        The function to simplify the source code

    Notes
    -----
    Currently this function generates and prunes source code in memory.
    It might be a problem if the program size is large.
    """

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

    queues = {} # signature(string) -> SimpleQueue[Program]
    workers = set()

    functions_dsl = [to_function(f) for f in functions]

    # Enumerate source code
    cnt = 0
    for program in source_code(functions_dsl, spec.min_program_length, spec.max_program_length):
        program = simplify_and_normalize(program) # Simplify the program
        if not (spec.min_program_length <= len(program.body) <= spec.max_program_length):
            # If the length of simplified program is out of range, discard the program
            continue

        signature = signature_to_string(get_signature(program))
        if not signature in queues:
            queues[signature] = mp.SimpleQueue()
            w = mp.Process(target=worker, args=(functions_dsl, spec, destinationDir, signature, queues[signature]))
            w.start()
            workers.add(w)

        # Enqueue the program to the queue
        cnt += 1
        queues[signature].put(program)

    for queue in queues.values():
        queue.put(None)
    for w in workers:
        w.join()
