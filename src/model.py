import dataclasses
import numpy as np
import chainer as ch
from chainer import link
import chainer.links as L
import chainer.functions as F
from chainer import backend
from chainer import reporter
from typing import List, Union, Dict
from src.dataset import DatasetStats


@dataclasses.dataclass
class ModelShapeParameters:
    dataset_stats: DatasetStats
    value_range: int
    max_list_length: int
    num_hidden_layers: int
    n_embed: int
    n_units: int


def weighted_sigmoid_cross_entropy(y, t, w_0: float = 0.5):
    """
    Compute weighted sigmoid cross entropy

    Parameters
    ----------
    y
        The prediction vector
    t
        The ground truth label
    w_0 : float
        The weight for label=0.
        If this value is negative, this function computes the original sigmoid cross entropy

    Returns
    -------
    ch.Variable
        computed cross entropy
    """
    if w_0 < 0:
        return F.sigmoid_cross_entropy(y, t)

    t_0 = t.copy()
    t_1 = t.copy()
    t_0[t_0 == 1] = -1
    t_1[t_0 == 0] = -1
    l0 = F.sigmoid_cross_entropy(y, t_0)
    l1 = F.sigmoid_cross_entropy(y, t_1)
    return w_0 * l0 + (1.0 - w_0) * l1


def tupled_binary_accuracy(y, t):
    """
    Compte binary classification accuracy

    Attributes
    ----------
    y
        The output predictions
    t
        The ground truth label
    """

    t_0 = t.copy()
    t_1 = t.copy()
    t_0[t_0 == 1] = -1
    t_1[t_0 == 0] = -1
    acc_0 = F.binary_accuracy(y, t_0)
    acc_1 = F.binary_accuracy(y, t_1)

    return acc_0, acc_1


class ExampleEmbed(link.Chain):
    """
    The embeded link of DeepCoder
    """

    def __init__(self, num_inputs: int, value_range: int, n_embed: int,
                 initialW: Union[None, np.array, ch.Initializer] = None):
        """
        Constructor

        Parameters
        ----------
        num_inputs : int
            The largest number of the inputs
        value_range : int
            The largest absolute value used in the dataset.
        n_embed : int
            The dimension of integer embedding. 20 was used in the paper.
        initialW : ch.Initializer or np.array or None
            The initial value of the weights
        """
        super(ExampleEmbed, self).__init__()

        with self.init_scope():
            self._embed_integer = L.EmbedID(
                2 * value_range + 1, n_embed, initialW=initialW)
        self._value_range = value_range
        self._num_inputs = num_inputs

    def forward(self, examples: np.array):
        """
        Computes the hidden layer encoding

        Parameters
        ----------
        examples : np.array
            Each element contains the list of PrimitiveExample

        Returns
        -------
        chainer.Variable
            The hidden layer encoding. The shape is (N, e, (num_inputs + 1), 2 + max_list_length * n_embed)
            where
                N is the minibatch size,
                e is the number of examples,
                num_inputs is the largest number of the inputs,
                max_list_length is the length of value encoding, and
                n_embed is the dimension of integer embedding.
        """

        N = examples.shape[0]  # minibatch size
        e = max(map(lambda x: len(x), examples))  # num of I/O examples
        num_inputs = self._num_inputs
        max_list_length = max(
            map(lambda x: x[0].inputs[0].value_arr.size, examples))  # max_list_length

        # Concatenate all values (inputs and output)
        # The one-hot array of type integers (N, e, (num_inputs + 1), 2)
        types = np.zeros((N, e, num_inputs + 1, 2), dtype=np.float32)
        for i, example_encodings in enumerate(examples):
            for j, example in enumerate(example_encodings):
                if len(example.inputs) > num_inputs:
                    raise RuntimeError("The number of inputs ({}) exceeds the limits ({})".format(
                        len(example.inputs), num_inputs))
                types[i, j, :len(example.inputs), :] = np.array(
                    list(map(lambda x: np.identity(2)[x.t], example.inputs)))
                types[i, j, num_inputs, :] = np.identity(2)[example.output.t]

        # The array of type values (N, e, num_inputs + 1, max_list_length)
        values = np.ones((N, e, self._num_inputs + 1, max_list_length),
                         dtype=np.int32) * (2 * self._value_range)
        for i, example_encodings in enumerate(examples):
            for j, example in enumerate(example_encodings):
                values[i, j, :len(example.inputs), :] = np.array(
                    list(map(lambda x: x.value_arr, example.inputs)))
                values[i, j, num_inputs, :] = example.output.value_arr

        # Convert the integer into the learned embeddings
        # (N, e, (num_inputs + 1), max_list_length, n_embed)
        values_embeddings = self._embed_integer(values)

        # Concat types and values
        n_embed = values_embeddings.shape[4]
        # (N, e, (num_inputs + 1), max_list_length * n_embed)
        values_embeddings = F.reshape(
            values_embeddings, (N, e, num_inputs + 1, -1))
        # (N, e, (num_inputs + 1), 2 + max_list_length * n_embed)
        state_embeddings = F.concat([types, values_embeddings], axis=3)

        return state_embeddings


class Encoder(link.Chain):
    """
    The encoder neural network of DeepCoder
    """

    def __init__(self, n_units: int,
                 num_hidden_layers: int = 3,
                 initialW: Union[None, ch.Initializer] = None,
                 initial_bias: Union[None, ch.Initializer] = None):
        """
        Constructor

        Parameters
        ----------
        n_units : int
            The number of units in the hidden layers. 256 was used in the paper.
        num_hidden_layers : int
            The number of hidden layers. 3 was used in the paper.
        initialW : ch.Initializer or None
            The initial value of the weights
        initial_bias : ch.Initializer or None
            The initial value of the biases
        """
        super(Encoder, self).__init__()

        linears = []
        with self.init_scope():
            layer = ch.Sequential(
                L.Linear(n_units, initialW=initialW,
                         initial_bias=initial_bias),
                F.sigmoid
            )
            self._hidden = layer.repeat(num_hidden_layers)

    def forward(self, state_embeddings: np.array):
        """
        Computes the hidden layer encoding

        Parameters
        ----------
        state_embeddings : np.array
            The state embeddings of the examples.
            The shape is (N, e, (num_inputs + 1), 2 + max_list_length * n_embed).

        Returns
        -------
        chainer.Variable
            The hidden layer encoding. The shape is (N, e, n_units)
            where
                N is the minibatch size,
                e is the number of examples, and
                n_unit is the number of units in the hidden layers.
        """

        N = state_embeddings.shape[0]  # minibatch size
        e = state_embeddings.shape[1]

        # Compute the hidden layer encoding
        # (N * e, (num_inputs + 1) * (2 + max_list_length * n_embed))
        state_embeddings = F.reshape(state_embeddings, (N * e, -1))
        output = self._hidden(state_embeddings)  # (N * e, n_units)
        output = F.reshape(output, (N, e, -1))
        return output


def Decoder(n_functions: int, initialW: Union[None, ch.Initializer, np.array] = None,
            initial_bias: Union[None, ch.Initializer, np.array] = None):
    """
    Returns the decoder of DeepCoder

    Parameters
    ----------
    n_functions : int
        The number of functions

    Returns
    -------
    chainer.Link
        The decoder of DeepCoder.
    """
    return ch.Sequential(
        # Input: (N, e, n_units)
        lambda x: F.mean(x, axis=1),
        # Pooled: (N, n_units)
        L.Linear(n_functions, initialW=initialW, initial_bias=initial_bias),
        # (N, n_functions)
    )


def Predictor(params: ModelShapeParameters) -> ch.Link:
    """
    Return the model of DeepCoder

    Parameters
    ----------
    params : ModelShapeParameters

    Returns
    -------
    ch.Link
        The model of DeepCoder
    """
    embed = ExampleEmbed(params.dataset_stats.max_num_inputs,
                         params.value_range, params.n_embed)
    encoder = Encoder(
        params.n_units, num_hidden_layers=params.num_hidden_layers)
    decoder = Decoder(len(params.dataset_stats.names))

    return ch.Sequential(embed, encoder, decoder)


def TrainingClassifier(predictor: ch.Link, w_0: float = -1):
    """
    Return the classifier for training DeepCoder

    Parameters
    ----------
    predictor : ch.Link
    w_0 : float
        The weight for label=False

    Returns
    -------
    chainer.Link
        The classifier used for training
    """

    classifier = L.Classifier(
        predictor,
        lossfun=lambda y, t: weighted_sigmoid_cross_entropy(y, t, w_0),
        accfun=F.binary_accuracy
    )

    def accuracy(y, t):
        acc_0, acc_1 = tupled_binary_accuracy(y, t)
        reporter.report(
            {"accuracy_false": acc_0, "accuracy_true": acc_1}, classifier)
        return F.binary_accuracy(y, t)
    classifier.accfun = accuracy
    return classifier
