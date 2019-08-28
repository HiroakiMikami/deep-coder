import numpy as np
import unittest
import chainer as ch
import chainer.functions as F

from src.model import ExampleEmbed, Encoder, Decoder, TrainingClassifier, tupled_binary_accuracy
from src.dataset import Example, examples_encoding, DatasetMetadata


class Test_model(unittest.TestCase):
    def test_example_embed_embed_one_sample(self):
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

        e = examples_encoding(
            [Example([[0, 1]], 0), Example([[1]], 1)],
            DatasetMetadata(1, set([]), 2, 2)
        )

        state_embeddings = embed.forward(np.array([e.types]), np.array([e.values]))
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
    def test_example_embed_embed_minibatch_with_different_number_of_inputs(self):
        embed = ExampleEmbed(2, 2, 1, (np.arange(5) + 1).reshape((5, 1)))
        """
        EmbedId
          0 (-2)   -> 1 
          1 (-1)   -> 2
          2 ( 0)   -> 3
          3 ( 1)   -> 4
          4 (NULL) -> 5
        """

        metadata = DatasetMetadata(2, set([]), 2, 2)
        e0 = examples_encoding([Example([[0, 1]], 0), Example([[1]], 1)], metadata)
        e1 = examples_encoding([Example([1, [0, 1]], [0]), Example([0, [0, 1]], [])], metadata)

        state_embeddings = embed.forward(np.array([e0.types, e1.types]), np.array([e0.values, e1.values]))
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

    def test_Encoder(self):
        embed = ExampleEmbed(1, 2, 1, (np.arange(5) + 1).reshape((5, 1)))

        encoder = Encoder(1, initialW=ch.initializers.One(),
                          initial_bias=ch.initializers.Zero())
        self.assertEqual(6, len(list(encoder.params())))
        """
        state_embeddings: (N, e, 2, 4) -> h1: (N, e, 1) -> h2: (N, e, 2) -> output: (N, e, 2)
        """

        metadata = DatasetMetadata(1, set([]), 2, 2)
        e = examples_encoding([Example([[0, 1]], 0), Example([[1]], 1)], metadata)

        state_embeddings = embed(np.array([e.types]), np.array([e.values]))
        layer_encodings = encoder(state_embeddings)

        self.assertEqual((1, 2, 1), layer_encodings.shape)
        for i in range(1):
            for j in range(2):
                h = np.array(state_embeddings[i, j, :, :].array.sum())
                h = F.sigmoid(F.sigmoid(F.sigmoid(h)))
                self.assertEqual(h.array, layer_encodings.array[i, j])

    def test_Decoder(self):
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

    def test_TrainingClassifier(self):
        embed = ExampleEmbed(1, 2, 2)
        encoder = Encoder(10)
        decoder = Decoder(2)
        classifier = TrainingClassifier(ch.Sequential(embed, encoder, decoder))

        metadata = DatasetMetadata(1, set([]), 2, 2)
        e = examples_encoding([Example([[0, 1]], 0), Example([[1]], 1)], metadata)
        labels = np.array([[1, 1]])
        loss = classifier(np.array([e.types]), np.array([e.values]), labels)
        loss.grad = np.ones(loss.shape, dtype=np.float32)

        # backward does not throw an error
        loss.backward()

    def test_tupled_binary_accuracy(self):
        acc = tupled_binary_accuracy(
            np.array([-1.0, -1.0, -1.0, 1.0]), np.array([0, 0, 1, 1]))
        self.assertAlmostEqual(1.0, acc[0].array)
        self.assertAlmostEqual(0.5, acc[1].array)


if __name__ == "__main__":
    unittest.main()
