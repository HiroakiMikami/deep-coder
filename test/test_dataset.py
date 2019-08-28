import unittest
import chainer as ch
import numpy as np

from src.dataset import Example, Entry, prior_distribution, primitive_encoding, attribute_encoding, examples_encoding, EncodedDataset, dataset_metadata, DatasetMetadata, Dataset


class Test_dataset(unittest.TestCase):
    def test_dataset_metadata(self):
        e0 = Entry("HEAD", [Example([[10, 20]], 10)], dict(
            [["HEAD", True], ["TAKE", False]]))
        e1 = Entry("TAKE", [Example([1, [10, 20]], 10)], dict(
            [["HEAD", False], ["TAKE", True]]))
        dataset = ch.datasets.TupleDataset([e0, e1])
        stats = dataset_metadata(dataset)
        self.assertEqual(2, stats.max_num_inputs)
        self.assertEqual(set(["HEAD", "TAKE"]), stats.symbols)

    def test_prior_distribution(self):
        dataset = ch.datasets.TupleDataset([
            Entry("", [], dict([["F1", True], ["F2", False]])),
            Entry("", [], dict([["F1", True], ["F2", True]]))
        ])
        prior = prior_distribution(dataset)
        self.assertAlmostEqual(1.0, prior["F1"])
        self.assertAlmostEqual(0.5, prior["F2"])

    def test_primitive_encoding(self):
        encoding = primitive_encoding(-10, DatasetMetadata(0, set([]), 256, 2))
        self.assertEqual(0, encoding.t)
        self.assertTrue(
            np.all(np.array([-10 + 256, 512] == encoding.value_arr)))

        encoding = primitive_encoding([1, 2], DatasetMetadata(0, set([]), 256, 3))
        self.assertEqual(1, encoding.t)
        self.assertTrue(
            np.all(np.array([257, 258, 512] == encoding.value_arr)))

    def test_attribute_encoding(self):
        encoding = attribute_encoding(dict([
            ["A", True],
            ["B", False]]))
        self.assertTrue(np.all(np.array([1, 0]) == encoding))

    def test_examples_encoding_if_num_inputs_is_too_large(self):
        metadata = DatasetMetadata(0, set([]), 2, 2)
        self.assertRaises(RuntimeError, lambda: examples_encoding([Example([1, [0, 1]], [0]), Example([0, [0, 1]], [])], metadata))

    def test_EncodedDataset_constructor(self):
        dataset = ch.datasets.TupleDataset([
            Entry("entry1", [Example(([10, 20, 30],), 10)],
                  dict([["HEAD", True], ["SORT", False]])),
            Entry(
                "entry2",
                [Example(([30, 20, 10],), [10, 20, 30])],
                dict([["HEAD", False], ["SORT", True]])
            )
        ])

        cdataset = EncodedDataset(Dataset(dataset, DatasetMetadata(1, set(["HEAD", "SORT"]), 256, 5)))
        [(examples0, attribute0), (examples1, attribute1)] = list(cdataset)

        self.assertTrue(np.all([[[0, 1], [1, 0]]] == examples0.types))
        self.assertTrue(
            np.all([[
                [266, 276, 286, 512, 512],
                [266, 512, 512, 512, 512]
            ]] == examples0.values))
        self.assertTrue(np.all(np.array([1, 0]) == attribute0))

        self.assertTrue(np.all([[[0, 1], [0, 1]]] == examples1.types))
        self.assertTrue(
            np.all([[
                [286, 276, 266, 512, 512],
                [266, 276, 286, 512, 512]
            ]] == examples1.values))
        self.assertTrue(np.all(np.array([0, 1]) == attribute1))


if __name__ == "__main__":
    unittest.main()
