import unittest

from src.dsl import Function, Type
from src.dataset import Entry, Dataset
import src.train as T

class Test_train(unittest.TestCase):
    def test_dataset_stats(self):
        HEAD = Function("HEAD", ([Type.IntList], Type.Int))
        TAKE = Function("TAKE", ([Type.Int, Type.IntList], Type.IntList))
        e0 = Entry("HEAD", [([[10, 20]], 10)], dict([[HEAD, True], [TAKE, False]]))
        e1 = Entry("TAKE", [([1, [10, 20]], 10)], dict([[HEAD, False], [TAKE, True]]))
        dataset = Dataset([e0, e1])
        stats = T.dataset_stats(dataset)
        self.assertEqual(2, stats.max_num_inputs)
        self.assertEqual(2, stats.num_functions)

if __name__ == "__main__":
    unittest.main()
