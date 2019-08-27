import unittest

from src.dataset import Entry, Dataset, prior_distribution, divide


class Test_prior_distribution(unittest.TestCase):
    def test_prior_distribution(self):
        dataset = Dataset([
            Entry("", [], dict([["F1", True], ["F2", False]])),
            Entry("", [], dict([["F1", True], ["F2", True]]))
        ])
        prior = prior_distribution(dataset)
        self.assertAlmostEqual(1.0, prior["F1"])
        self.assertAlmostEqual(0.5, prior["F2"])


class Test_divide(unittest.TestCase):
    def test_divide(self):
        dataset = Dataset([
            Entry("e1", [], dict([["F1", True]])),
            Entry("e2", [], dict([["F1", True]])),
            Entry("e3", [], dict([["F1", True]])),
            Entry("e4", [], dict([["F1", True]])),
        ])
        d = divide(dataset, dict([["train", 2], ["dev", 1], ["valid", 1]]))
        self.assertEqual(2, len(d["train"].entries))
        self.assertEqual(1, len(d["dev"].entries))
        self.assertEqual(1, len(d["valid"].entries))
        self.assertFalse(d["train"].entries[0].source_code ==
                         d["train"].entries[1].source_code)
        self.assertFalse(d["train"].entries[0].source_code ==
                         d["dev"].entries[0].source_code)
        self.assertFalse(d["train"].entries[1].source_code ==
                         d["dev"].entries[0].source_code)
        self.assertFalse(d["dev"].entries[0].source_code ==
                         d["valid"].entries[0].source_code)


if __name__ == "__main__":
    unittest.main()
