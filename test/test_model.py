import numpy as np
import unittest
import chainer as ch
import chainer.functions as F

from src.model import ExampleEmbed, Encoder, Decoder, TrainingClassifier, tupled_binary_accuracy
from src.chainer_dataset import ExampleEncoding, encode_example


class TestExampleEmbed(unittest.TestCase):
    def test_embed_one_sample(self):
        embed = ExampleEmbed(1, 2, 1, (np.arange(5) + 1).reshape((5, 1)))
        self.assertEqual(1, len(list(embed.params())))
        """
        EmbedId
          0 (-2)   -> 1 
          1 (-1)   -> 2
          2 ( 0)   -> 3
          3 ( 1)   -> 4
          4 (NULL) -> 5
        """

        e1 = encode_example(([[0, 1]], 0), 2, 2)
        e2 = encode_example(([[1]], 1), 2, 2)
        minibatch = np.array([[e1, e2]])

        N = 1
        e = 2
        I = 1
        value_range = 2
        max_list_length = 2

        state_embeddings = embed.forward(minibatch)
        self.assertEqual((1, 2, 2, 2 + 2 * 1), state_embeddings.shape)
        self.assertTrue(np.allclose(
            [0, 1, 3, 4], state_embeddings.array[0, 0, 0]))  # Input of e1
        self.assertTrue(np.allclose(
            [1, 0, 3, 5], state_embeddings.array[0, 0, 1]))  # Output of e1
        self.assertTrue(np.allclose(
            [0, 1, 4, 5], state_embeddings.array[0, 1, 0]))  # Input of e2
        self.assertTrue(np.allclose(
            [1, 0, 4, 5], state_embeddings.array[0, 1, 1]))  # Output of e2

        # backward does not throw an error
        state_embeddings.grad = np.ones(
            state_embeddings.shape, dtype=np.float32)
        state_embeddings.backward()

    # minibatch with mask
    def test_embed_minibatch_with_different_number_of_inputs(self):
        embed = ExampleEmbed(2, 2, 1, (np.arange(5) + 1).reshape((5, 1)))
        """
        EmbedId
          0 (-2)   -> 1 
          1 (-1)   -> 2
          2 ( 0)   -> 3
          3 ( 1)   -> 4
          4 (NULL) -> 5
        """

        e00 = encode_example(([[0, 1]], 0), 2, 2)
        e01 = encode_example(([[1]], 1), 2, 2)
        e10 = encode_example(([1, [0, 1]], [0]), 2, 2)
        e11 = encode_example(([0, [0, 1]], []), 2, 2)
        minibatch = np.array([[e00, e01], [e10, e11]])

        N = 1
        e = 2
        I = 2
        value_range = 2
        max_list_length = 2

        state_embeddings = embed.forward(minibatch)
        self.assertEqual((2, 2, 3, 2 + 2 * 1), state_embeddings.shape)
        self.assertTrue(np.allclose(
            [0, 1, 3, 4], state_embeddings.array[0, 0, 0]))  # Input of e00
        self.assertTrue(np.allclose(
            [0, 0, 5, 5], state_embeddings.array[0, 0, 1]))  # Input of e00
        # Output of e00
        self.assertTrue(np.allclose(
            [1, 0, 3, 5], state_embeddings.array[0, 0, 2]))
        self.assertTrue(np.allclose(
            [0, 1, 4, 5], state_embeddings.array[0, 1, 0]))  # Input of e01
        self.assertTrue(np.allclose(
            [0, 0, 5, 5], state_embeddings.array[0, 1, 1]))  # Input of e01
        # Output of e01
        self.assertTrue(np.allclose(
            [1, 0, 4, 5], state_embeddings.array[0, 1, 2]))
        self.assertTrue(np.allclose(
            [1, 0, 4, 5], state_embeddings.array[1, 0, 0]))  # Input of e10
        self.assertTrue(np.allclose(
            [0, 1, 3, 4], state_embeddings.array[1, 0, 1]))  # Input of e10
        # Output of e10
        self.assertTrue(np.allclose(
            [0, 1, 3, 5], state_embeddings.array[1, 0, 2]))
        self.assertTrue(np.allclose(
            [1, 0, 3, 5], state_embeddings.array[1, 1, 0]))  # Input of e11
        self.assertTrue(np.allclose(
            [0, 1, 3, 4], state_embeddings.array[1, 1, 1]))  # Input of e11
        # Output of e11
        self.assertTrue(np.allclose(
            [0, 1, 5, 5], state_embeddings.array[1, 1, 2]))

    def test_throw_error_if_num_inputs_is_too_large(self):
        embed = ExampleEmbed(1, 2, 1, (np.arange(5) + 1).reshape((5, 1)))
        e0 = encode_example(([1, [0, 1]], [0]), 2, 2)
        e1 = encode_example(([0, [0, 1]], []), 2, 2)
        minibatch = np.array([[e0, e1]])

        self.assertRaises(RuntimeError, lambda: embed(minibatch))


class TestEncoder(unittest.TestCase):
    def test_encoder(self):
        embed = ExampleEmbed(1, 2, 1, (np.arange(5) + 1).reshape((5, 1)))

        encoder = Encoder(1, initialW=ch.initializers.One(),
                          initial_bias=ch.initializers.Zero())
        self.assertEqual(6, len(list(encoder.params())))
        """
        state_embeddings: (N, e, 2, 4) -> h1: (N, e, 1) -> h2: (N, e, 2) -> output: (N, e, 2)
        """

        e1 = encode_example(([[0, 1]], 0), 2, 2)
        e2 = encode_example(([[1]], 1), 2, 2)
        minibatch = np.array([[e1, e2]])

        state_embeddings = embed(minibatch)
        layer_encodings = encoder(state_embeddings)

        self.assertEqual((1, 2, 1), layer_encodings.shape)
        for i in range(1):
            for j in range(2):
                h = np.array(state_embeddings[i, j, :, :].array.sum())
                h = F.sigmoid(F.sigmoid(F.sigmoid(h)))
                self.assertEqual(h.array, layer_encodings.array[i, j])


class TestDecoder(unittest.TestCase):
    def test_decoder(self):
        initialW = np.ones((1, 2))
        initial_bias = np.zeros((1,))

        decoder = Decoder(1, ch.initializers.One(), ch.initializers.Zero())
        self.assertEqual(2, len(list(decoder.params())))

        input = np.zeros((1, 2, 2), dtype=np.float32)
        input[0, 1, :] = 1.0
        output = decoder(input)
        """
        [[0, 0], [1, 1]] =(pool)> [[0.5, 0.5]] =(linear)> [1] -> sigmoid
        """

        self.assertEqual((1, 1), output.shape)
        self.assertTrue(np.allclose(np.array([1.0]), output.array))


class TestTrainingClassifier(unittest.TestCase):
    def test_training_classifier(self):
        embed = ExampleEmbed(1, 2, 2)
        encoder = Encoder(10)
        decoder = Decoder(2)
        classifier = TrainingClassifier(embed, encoder, decoder)

        e1 = encode_example(([[0, 1]], 0), 2, 2)
        e2 = encode_example(([[1]], 1), 2, 2)
        minibatch = np.array([[e1, e2]])
        labels = np.array([[1, 1]])
        loss = classifier(minibatch, labels)
        loss.grad = np.ones(loss.shape, dtype=np.float32)

        # backward does not throw an error
        loss.backward()


class Test_tupled_binary_accuracy(unittest.TestCase):
    def test_tupled_binary_accuracy(self):
        acc = tupled_binary_accuracy(
            np.array([-1.0, -1.0, -1.0, 1.0]), np.array([0, 0, 1, 1]))
        self.assertAlmostEqual(1.0, acc[0].array)
        self.assertAlmostEqual(0.5, acc[1].array)


if __name__ == "__main__":
    unittest.main()
