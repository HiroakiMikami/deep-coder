import unittest
import os
import numpy as np
import chainer as ch

from src.dataset import Entry, Example, encode_example, DatasetStats
from src.deepcoder_utils import generate_io_samples
from src.model import ModelShapeParameters
from src.inference import search, predict_with_prior_distribution, predict_with_neural_network, InferenceModel


class Test_inferense(unittest.TestCase):
    def test_search(self):
        # example of access
        examples = [
            Example([2, [10, 20, 30]], 30),
            Example([1, [-10, 30, 40]], 30)
        ]

        def pred(examples):
            LINQ, _ = generate_io_samples.get_language(50)
            LINQ = [f for f in LINQ if not "IDT" in f.src]
            prob = dict()
            for function in LINQ:
                for name in function.src.split(" "):
                    if name == "ACCESS":
                        prob[name] = 0.8
                    else:
                        prob[name] = 0.2
            return prob

        result = search(
            os.path.join(os.getcwd(), "DeepCoder_Utils",
                         "enumerative-search", "search"), 1000, 256,
            examples, 2, pred)

        self.assertTrue(result.is_solved)
        self.assertAlmostEqual(0.8, result.probabilities["ACCESS"])
        self.assertAlmostEqual(0.2, result.probabilities["HEAD"])
        self.assertEqual(1, result.explored_nodes)
        self.assertEqual(" %2 <- access %0 %1\n", result.solution)

    def test_search_when_pred_throws_error(self):
        # example that do not correspond to any programs
        examples = [
            Example([2, [10, 20, 30]], -255),
            Example([1, [-10, 30, 40]], -255)
        ]

        def pred(examples):
            raise RuntimeError("test")

        result = search(
            os.path.join(os.getcwd(), "DeepCoder_Utils",
                         "enumerative-search", "search"), 1000, 256,
            examples, 2, pred)

        self.assertFalse(result.is_solved)
        self.assertEqual(-1, result.explored_nodes)
        self.assertEqual(dict([]), result.probabilities)
        self.assertEqual("", result.solution)

    def test_search_with_invalid_examples(self):
        # example that do not correspond to any programs
        examples = [
            Example([2, [10, 20, 30]], -255),
            Example([1, [-10, 30, 40]], -255)
        ]

        def pred(examples):
            LINQ, _ = generate_io_samples.get_language(50)
            LINQ = [f for f in LINQ if not "IDT" in f.src]
            prob = dict()
            for function in LINQ:
                for name in function.src.split(" "):
                    prob[name] = 1.0
            return prob

        result = search(
            os.path.join(os.getcwd(), "DeepCoder_Utils",
                         "enumerative-search", "search"), 1000, 256,
            examples, 2, pred)

        self.assertFalse(result.is_solved)
        self.assertEqual(-1, result.explored_nodes)
        self.assertEqual("", result.solution)

    def test_predict_with_prior_distribution(self):
        dataset = ch.datasets.TupleDataset([
            Entry("e0", [], dict([["MAP", True], ["HEAD", True]])),
            Entry("e1", [], dict([["MAP", False], ["HEAD", True]]))
        ])
        pred = predict_with_prior_distribution(dataset)
        prob = pred([])
        self.assertEqual(dict([["MAP", 0.5], ["HEAD", 1.0]]), prob)

    def test_predict_with_neural_network(self):
        examples = [
            Example([2, [10, 20, 30]], 30),
            Example([1, [-10, 30, 40]], 30)
        ]
        model_shape = ModelShapeParameters(DatasetStats(
            2, set(["MAP", "HEAD"])), 256, 5, 3, 2, 10)
        m = InferenceModel(model_shape)
        pred = predict_with_neural_network(model_shape, m)
        prob = pred(examples)

        example_encodings = np.array(
            [[encode_example(example, 256, 5) for example in examples]])
        prob_dnn = m.model(example_encodings).array[0]

        self.assertAlmostEqual(prob_dnn[0], prob["HEAD"])
        self.assertAlmostEqual(prob_dnn[1], prob["MAP"])


if __name__ == "__main__":
    unittest.main()
