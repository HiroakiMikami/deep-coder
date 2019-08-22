import unittest
import tempfile
import pickle
import os
import numpy as np
from src.deepcoder_utils import generate_io_samples
from src.dsl import Function, Type, Variable, Expression, Program, to_string, clone
from src.generate_dataset import generate_dataset, DatasetSpec, EquivalenceCheckingSpec
from src.source_code_simplifier import remove_redundant_variables

class Test_generate_dataset(unittest.TestCase):
    def test_generate_dataset(self):
        LINQ, _ = generate_io_samples.get_language(50)
        HEAD = [f for f in LINQ if f.src == "HEAD"][0]
        TAKE = [f for f in LINQ if f.src == "TAKE"][0]

        # Generate the program with the length of 1
        with tempfile.TemporaryDirectory() as name:
            generate_dataset([HEAD, TAKE], DatasetSpec(50, 20, 5, 1, 1), EquivalenceCheckingSpec(1.0, 1, None), name)
            # Check the dataset
            srcs = set()
            for p in os.listdir(name):
                with open(os.path.join(name, p), "rb") as fp:
                    dataset = pickle.load(fp)
                    for entry in dataset.entries:
                        srcs.add(entry.source_code)
                        p = generate_io_samples.compile(entry.source_code, 50, 5)
                        self.assertNotEqual(None, p)
                        for example in entry.examples:
                            output = p.fun(example[0])
                            self.assertEqual(output, example[1])
            self.assertEqual(set(["a <- int\nb <- [int]\nc <- TAKE a b", "a <- [int]\nb <- HEAD a"]), srcs)

        # Generate the program with the length of 2
        with tempfile.TemporaryDirectory() as name:
            def simplify(program):
                program = remove_redundant_variables(program)
                return program
            generate_dataset([HEAD, TAKE], DatasetSpec(50, 20, 5, 2, 2), EquivalenceCheckingSpec(1.0, 1, None), name, simplify=simplify)

            # Check the dataset
            srcs = set()
            for p in os.listdir(name):
                with open(os.path.join(name, p), "rb") as fp:
                    dataset = pickle.load(fp)
                    for entry in dataset.entries:
                        srcs.add(entry.source_code)
                        p = generate_io_samples.compile(entry.source_code, 50, 5)
                        self.assertNotEqual(None, p)
                        for example in entry.examples:
                            output = p.fun(example[0])
                            self.assertEqual(output, example[1])
            self.assertEqual(set([
                "a <- [int]\nb <- HEAD a\nc <- TAKE b a",
                "a <- int\nb <- [int]\nc <- TAKE a b\nd <- TAKE a c",
                "a <- int\nb <- [int]\nc <- int\nd <- TAKE a b\ne <- TAKE c d",
                "a <- int\nb <- [int]\nc <- TAKE a b\nd <- HEAD c",
                "a <- [int]\nb <- [int]\nc <- HEAD a\nd <- TAKE c b"
            ]), srcs)

    def test_generate_dataset_can_relax_equivalence_checking(self):
        LINQ, _ = generate_io_samples.get_language(50)
        HEAD = [f for f in LINQ if f.src == "HEAD"][0]
        LAST = [f for f in LINQ if f.src == "LAST"][0]

        # Generate the program with the length of 1
        with tempfile.TemporaryDirectory() as name:
            np.random.seed(0)
            generate_dataset([HEAD, LAST], DatasetSpec(50, 20, 5, 1, 1), EquivalenceCheckingSpec(0, 1, None), name)
            # Check the dataset
            srcs = set()
            for p in os.listdir(name):
                with open(os.path.join(name, p), "rb") as fp:
                    dataset = pickle.load(fp)
                    for entry in dataset.entries:
                        srcs.add(entry.source_code)
                        p = generate_io_samples.compile(entry.source_code, 50, 5)
                        self.assertNotEqual(None, p)
                        for example in entry.examples:
                            output = p.fun(example[0])
                            self.assertEqual(output, example[1])
            self.assertEqual(set(["a <- [int]\nb <- HEAD a", "a <- [int]\nb <- LAST a"]), srcs)

if __name__ == "__main__":
    unittest.main()
