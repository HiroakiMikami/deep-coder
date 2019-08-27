import unittest
import chainer as ch
import numpy as np

from src.dataset import Example, Entry, prior_distribution, encode_primitive, encode_attribute, EncodedDataset, dataset_stats


class Test_dataset_stats(unittest.TestCase):
    def test_dataset_stats(self):
        e0 = Entry("HEAD", [Example([[10, 20]], 10)], dict(
            [["HEAD", True], ["TAKE", False]]))
        e1 = Entry("TAKE", [Example([1, [10, 20]], 10)], dict(
            [["HEAD", False], ["TAKE", True]]))
        dataset = ch.datasets.TupleDataset([e0, e1])
        stats = dataset_stats(dataset)
        self.assertEqual(2, stats.max_num_inputs)
        self.assertEqual(set(["HEAD", "TAKE"]), stats.names)


class Test_prior_distribution(unittest.TestCase):
    def test_prior_distribution(self):
        dataset = ch.datasets.TupleDataset([
            Entry("", [], dict([["F1", True], ["F2", False]])),
            Entry("", [], dict([["F1", True], ["F2", True]]))
        ])
        prior = prior_distribution(dataset)
        self.assertAlmostEqual(1.0, prior["F1"])
        self.assertAlmostEqual(0.5, prior["F2"])


class Test_encode_primitive(unittest.TestCase):
    def test_encode_primitive(self):
        encoding = encode_primitive(-10, 256, 2)
        self.assertEqual(0, encoding.t)
        self.assertTrue(
            np.all(np.array([-10 + 256, 512] == encoding.value_arr)))

        encoding = encode_primitive([1, 2], 256, 3)
        self.assertEqual(1, encoding.t)
        self.assertTrue(
            np.all(np.array([257, 258, 512] == encoding.value_arr)))


class Test_encode_attribute(unittest.TestCase):
    def test_encode_attribute(self):
        encoding = encode_attribute(dict([
            ["A", True],
            ["B", False]]))
        self.assertTrue(np.all(np.array([1, 0]) == encoding))


class Test_EncodedDataset(unittest.TestCase):
    def test_constructor(self):
        dataset = ch.datasets.TupleDataset([
            Entry("entry1", [Example(([10, 20, 30],), 10)],
                  dict([["HEAD", True], ["SORT", False]])),
            Entry(
                "entry2",
                [Example(([30, 20, 10],), [10, 20, 30])],
                dict([["HEAD", False], ["SORT", True]])
            )
        ])

        cdataset = EncodedDataset(dataset, 256, 5)
        [(example0, attribute0), (example1, attribute1)] = list(cdataset)

        self.assertEqual(1, example0[0].inputs[0].t)
        self.assertTrue(
            np.all(np.array([266, 276, 286, 512, 512]) == example0[0].inputs[0].value_arr))
        self.assertEqual(0, example0[0].output.t)
        self.assertTrue(
            np.all(np.array([266, 512, 512, 512, 512]) == example0[0].output.value_arr))
        self.assertTrue(np.all(np.array([1, 0]) == attribute0))

        self.assertEqual(1, example1[0].inputs[0].t)
        self.assertTrue(
            np.all(np.array([286, 276, 266, 512, 512]) == example1[0].inputs[0].value_arr))
        self.assertEqual(1, example1[0].output.t)
        self.assertTrue(
            np.all(np.array([266, 276, 286, 512, 512]) == example1[0].output.value_arr))
        self.assertTrue(np.all(np.array([0, 1]) == attribute1))


if __name__ == "__main__":
    unittest.main()
