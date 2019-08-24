import unittest
import numpy as np

from src.dsl import Function, Type
from src.dataset import Dataset, Entry
from src.chainer_dataset import encode_primitive, encode_attribute, PrimitiveEncoding, ChainerDataset

class Test_encode_primitive(unittest.TestCase):
    def test_encode_primitive(self):
        encoding = encode_primitive(-10, 256, 2)
        self.assertEqual(0, encoding.t)
        self.assertTrue(np.all(np.array([-10 + 256, 512] == encoding.value_arr)))

        encoding = encode_primitive([1, 2], 256, 3)
        self.assertEqual(1, encoding.t)
        self.assertTrue(np.all(np.array([257, 258, 512] == encoding.value_arr)))

class Test_encode_attribute(unittest.TestCase):
    def test_encode_attribute(self):
        encoding = encode_attribute(dict([
            [Function("A", ([Type.Int], Type.Int)), True],
            [Function("B", ([Type.Int], Type.Int)), False]]))
        self.assertTrue(np.all(np.array([1, 0]) == encoding))

class Test_ChainerDataset(unittest.TestCase):
    def test_constructor(self):
        HEAD = Function("HEAD", ([Type.IntList], Type.Int))
        SORT = Function("SORT", ([Type.IntList], Type.IntList))

        dataset = Dataset([])
        dataset.entries.append(Entry(
            "entry1",
            [(([10, 20, 30],), 10)],
            dict([[HEAD, True], [SORT, False]])
        ))
        dataset.entries.append(Entry(
            "entry2",
            [(([30, 20, 10],), [10, 20, 30])],
            dict([[HEAD, False], [SORT, True]])
        ))

        cdataset = ChainerDataset(dataset, 256, 5)
        [(example0, attribute0), (example1, attribute1)] = list(cdataset)

        self.assertEqual(1, example0[0].inputs[0].t)
        self.assertTrue(np.all(np.array([266, 276, 286, 512, 512]) == example0[0].inputs[0].value_arr))
        self.assertEqual(0, example0[0].output.t)
        self.assertTrue(np.all(np.array([266, 512, 512, 512, 512]) == example0[0].output.value_arr))
        self.assertTrue(np.all(np.array([1, 0]) == attribute0))

        self.assertEqual(1, example1[0].inputs[0].t)
        self.assertTrue(np.all(np.array([286, 276, 266, 512, 512]) == example1[0].inputs[0].value_arr))
        self.assertEqual(1, example1[0].output.t)
        self.assertTrue(np.all(np.array([266, 276, 286, 512, 512]) == example1[0].output.value_arr))
        self.assertTrue(np.all(np.array([0, 1]) == attribute1))

if __name__ == "__main__":
    unittest.main()
