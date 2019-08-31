import unittest
import tempfile
import pickle
import os
import numpy as np
from src.deepcoder_utils import generate_io_samples
from src.dsl import Function, Type, Variable, Expression, Program
from src.dataset import DatasetMetadata
from src.generate_dataset import generate_dataset, DatasetSpec, EquivalenceCheckingSpec, IteratorDecorator
from src.program_simplifier import remove_redundant_variables


class Test_generate_dataset(unittest.TestCase):
    def test_generate_dataset(self):
        LINQ, _ = generate_io_samples.get_language(50)
        HEAD = [f for f in LINQ if f.src == "HEAD"][0]
        TAKE = [f for f in LINQ if f.src == "TAKE"][0]

        # Generate the program with the length of 1
        with tempfile.NamedTemporaryFile() as f:
            name = f.name
            generate_dataset([HEAD, TAKE], DatasetSpec(
                50, 20, 5, 1, 1), EquivalenceCheckingSpec(1.0, 1, None), name)
            # Check the dataset
            srcs = set()
            with open(name, "rb") as fp:
                d = pickle.load(fp)
                dataset = d.dataset
                metadata = d.metadata
                for entry, in dataset:
                    srcs.add(entry.source_code)
                    p = generate_io_samples.compile(
                        entry.source_code, 50, 5)
                    self.assertNotEqual(None, p)
                    for example in entry.examples:
                        output = p.fun(example.inputs)
                        self.assertEqual(output, example.output)
            self.assertEqual(
                set(["a <- int\nb <- [int]\nc <- TAKE a b", "a <- [int]\nb <- HEAD a"]), srcs)
            self.assertEqual(DatasetMetadata(
                2, set(["TAKE", "HEAD"]), 50, 20), metadata)

        # Generate the program with the length of 2
        with tempfile.NamedTemporaryFile() as f:
            name = f.name

            def simplify(program):
                program = remove_redundant_variables(program)
                return program
            generate_dataset([HEAD, TAKE], DatasetSpec(
                50, 20, 5, 2, 2), EquivalenceCheckingSpec(1.0, 1, None), name, simplify=simplify)

            # Check the dataset
            srcs = set()
            with open(name, "rb") as fp:
                d = pickle.load(fp)
                dataset = d.dataset
                metadata = d.metadata
                for entry, in dataset:
                    srcs.add(entry.source_code)
                    p = generate_io_samples.compile(
                        entry.source_code, 50, 5)
                    self.assertNotEqual(None, p)
                    for example in entry.examples:
                        output = p.fun(example.inputs)
                        self.assertEqual(output, example.output)
            self.assertEqual(set([
                "a <- [int]\nb <- HEAD a\nc <- TAKE b a",
                "a <- int\nb <- [int]\nc <- TAKE a b\nd <- TAKE a c",
                "a <- int\nb <- [int]\nc <- int\nd <- TAKE a b\ne <- TAKE c d",
                "a <- int\nb <- [int]\nc <- TAKE a b\nd <- HEAD c",
                "a <- [int]\nb <- [int]\nc <- HEAD a\nd <- TAKE c b"
            ]), srcs)
            self.assertEqual(DatasetMetadata(
                3, set(["TAKE", "HEAD"]), 50, 20), metadata)

    def test_generate_dataset_can_relax_equivalence_checking(self):
        LINQ, _ = generate_io_samples.get_language(50)
        HEAD = [f for f in LINQ if f.src == "HEAD"][0]
        LAST = [f for f in LINQ if f.src == "LAST"][0]

        # Generate the program with the length of 1
        with tempfile.NamedTemporaryFile() as f:
            name = f.name
            np.random.seed(0)
            generate_dataset([HEAD, LAST], DatasetSpec(
                50, 20, 5, 1, 1), EquivalenceCheckingSpec(0, 1, None), name)
            # Check the dataset
            srcs = set()
            with open(name, "rb") as fp:
                d = pickle.load(fp)
                dataset = d.dataset
                metadata = d.metadata
                for entry, in dataset:
                    srcs.add(entry.source_code)
                    p = generate_io_samples.compile(
                        entry.source_code, 50, 5)
                    self.assertNotEqual(None, p)
                    for example in entry.examples:
                        output = p.fun(example.inputs)
                        self.assertEqual(output, example.output)
            self.assertEqual(
                set(["a <- [int]\nb <- HEAD a", "a <- [int]\nb <- LAST a"]), srcs)
            self.assertEqual(DatasetMetadata(
                1, set(["HEAD", "LAST"]), 50, 20), metadata)

    def test_generate_dataset_with_decorators(self):
        LINQ, _ = generate_io_samples.get_language(50)
        HEAD = [f for f in LINQ if f.src == "HEAD"][0]
        LAST = [f for f in LINQ if f.src == "LAST"][0]

        class Decorator:
            def __init__(self):
                self.items = []
            def __call__(self, generator):
                for item in generator:
                    self.items.append(item)
                    yield item
        program = Decorator()
        entries = Decorator()
        decorator = IteratorDecorator(program, entries)

        # Generate the program with the length of 1
        with tempfile.NamedTemporaryFile() as f:
            name = f.name
            np.random.seed(0)
            generate_dataset([HEAD, LAST], DatasetSpec(
                50, 20, 5, 1, 1), EquivalenceCheckingSpec(1, 1, None), name, decorator=decorator)
            self.assertEqual(2, len(program.items))
            self.assertEqual(1, len(entries.items))

    def test_generate_dataset_separate_higher_order_function_and_lambda(self):
        LINQ, _ = generate_io_samples.get_language(50)
        HEAD = [f for f in LINQ if f.src == "HEAD"][0]
        MAP_INC = [f for f in LINQ if f.src == "MAP INC"][0]

        # Generate the program with the length of 1
        with tempfile.NamedTemporaryFile() as f:
            name = f.name
            np.random.seed(0)
            generate_dataset([HEAD, MAP_INC], DatasetSpec(
                50, 20, 5, 1, 1), EquivalenceCheckingSpec(1, 1, None), name)
            # Check the dataset
            attribute_keys = set()
            with open(name, "rb") as fp:
                d = pickle.load(fp)
                dataset = d.dataset
                metadata = d.metadata
                for entry, in dataset:
                    for symbol in entry.attribute.keys():
                        attribute_keys.add(symbol)
            self.assertEqual(set(["HEAD", "MAP", "INC"]), attribute_keys)
            self.assertEqual(DatasetMetadata(
                1, set(["HEAD", "MAP", "INC"]), 50, 20), metadata)

    def test_generate_dataset_fix_number_of_dataset(self):
        LINQ, _ = generate_io_samples.get_language(50)
        HEAD = [f for f in LINQ if f.src == "HEAD"][0]
        LAST = [f for f in LINQ if f.src == "LAST"][0]
        MAXIMUM = [f for f in LINQ if f.src == "MAXIMUM"][0]

        # Generate the program with the length of 1
        with tempfile.NamedTemporaryFile() as f:
            name = f.name
            np.random.seed(0)
            generate_dataset([HEAD, LAST, MAXIMUM], DatasetSpec(
                50, 20, 5, 1, 1), EquivalenceCheckingSpec(1, 1, None), name, 2)
            # Check the dataset
            attribute_keys = set()
            with open(name, "rb") as fp:
                d = pickle.load(fp)
                dataset = d.dataset
                metadata = d.metadata
            self.assertEqual(2, len(dataset))
            self.assertTrue(dataset[0][0].source_code != dataset[1][0].source_code)

if __name__ == "__main__":
    unittest.main()
