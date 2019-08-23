import unittest

from src.dsl import Function, Type
from src.dataset import Entry, Dataset, prior_distribution

class Test_prior_distribution(unittest.TestCase):
    def test_prior_distribution(self):
        F1 = Function("F1", ([Type.IntList], Type.Int))
        F2 = Function("F2", ([Type.IntList], Type.Int))
        dataset = Dataset([
            Entry("", [], dict([[F1, True], [F2, False]])),
            Entry("", [], dict([[F1, True], [F2, True]]))
        ])
        prior = prior_distribution(dataset)
        self.assertAlmostEqual(1.0, prior[F1])
        self.assertAlmostEqual(0.5, prior[F2])

if __name__ == "__main__":
    unittest.main()
