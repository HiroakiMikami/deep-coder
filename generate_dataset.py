import numpy as np
import random
import os
from tqdm import tqdm
import argparse
from src.dsl import to_function, Program
from src.deepcoder_utils import generate_io_samples
from src.generate_dataset import generate_dataset, DatasetSpec, EquivalenceCheckingSpec, ProgressCallback
from src.source_code_simplifier import remove_redundant_variables, remove_redundant_expressions, remove_dependency_between_variables

SEED_MAX = 2**32 - 1

parser = argparse.ArgumentParser()
parser.add_argument(
    "--value-range", help="The largest absolute value used in the dataset", type=int, default=256)
parser.add_argument("--max-list-length",
                    help="The maximum length of the list used in the dataset", type=int, default=20)
parser.add_argument(
    "--num-examples", help="The number of I/O examples per program", type=int, default=5)
parser.add_argument(
    "--min-length", help="The minimum length of the program body", type=int, default=1)
parser.add_argument(
    "--max-length", help="The maximum length of the program body", type=int, default=1)
parser.add_argument("--seed", help="The random seed", type=int, default=6217)
parser.add_argument("--num-examples-for-pruning",
                    help="The number of examples used to prune the identical programs", type=int, default=100)
parser.add_argument(
    "destination", help="The directory that will contain the dataset", type=str)
args = parser.parse_args()

root_rng = np.random.RandomState(args.seed)
random.seed(root_rng.randint(SEED_MAX))
np.random.seed(root_rng.randint(SEED_MAX))

LINQ, _ = generate_io_samples.get_language(args.value_range)
LINQ = [f for f in LINQ if not "IDT" in f.src]

MINIMUM = to_function([f for f in LINQ if f.src == "MINIMUM"][0])
MAXIMUM = to_function([f for f in LINQ if f.src == "MAXIMUM"][0])


def simplify(program):
    program = remove_redundant_expressions(program)
    program = remove_redundant_variables(program)
    program = remove_dependency_between_variables(program, MINIMUM, MAXIMUM)
    return program


class Tqdm:
    def __init__(self):
        self._tqdm_enumeration = tqdm(desc="Program Generation")
        self._tqdm_dump_dataset = None

    def on_generate_program(self, program: Program):
        self._tqdm_enumeration.update(1)

    def on_finish_enumeration(self, n_programs: int):
        self._tqdm_enumeration.close()
        self._tqdm_dump_dataset = tqdm(
            total=n_programs, desc="Dataset Generation")

    def on_dump_dataset(self, n: int):
        self._tqdm_dump_dataset.update(n)

    def finish(self):
        self._tqdm_dump_dataset.close()


tqdm_for_generation = Tqdm()
callback = ProgressCallback(lambda p: tqdm_for_generation.on_generate_program(
    p), lambda x: tqdm_for_generation.on_finish_enumeration(x), lambda x: tqdm_for_generation.on_dump_dataset(x))

if not os.path.exists(args.destination):
    os.makedirs(args.destination)
assert(os.path.isdir(args.destination))

generate_dataset(LINQ,
                 DatasetSpec(args.value_range, args.max_list_length,
                             args.num_examples, args.min_length, args.max_length),
                 EquivalenceCheckingSpec(0, args.num_examples_for_pruning, np.random.RandomState(
                     root_rng.randint(SEED_MAX))),
                 args.destination, simplify=simplify, callback=callback)
tqdm_for_generation.finish()
