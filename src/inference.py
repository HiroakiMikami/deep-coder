import dataclasses
import tempfile
import os
import numpy as np
import subprocess
import chainer as ch
import chainer.functions as F
from typing import List, Union, Dict, Callable, Set
from .dataset import Example, prior_distribution, examples_encoding
from .model import Predictor, ModelShapeParameters


class InferenceModel:
    """
    The model for inference

    Attributes
    ----------
    predictor : ch.Link
        The attribute predictor of DeepCoder
    model : ch.Link
        The model that outputs probabilities of each symbols
    """

    def __init__(self, params: ModelShapeParameters):
        """
        Constructor

        Parameters
        ----------
        params : ModelShapeParameters
        """
        self.predictor = Predictor(params)
        self.model = ch.Sequential(self.predictor, F.sigmoid)


@dataclasses.dataclass
class SearchResult:
    is_solved: bool
    probabilities: Dict[str, float]
    solution: str
    explored_nodes: int
    time_seconds: float


def search(search: str, timeout_second: int, value_range: int,
           examples: List[Example], max_program_length: int, pred: Callable[[List[Example]], Dict[str, float]]) -> SearchResult:
    """
    Search over program space and return the result of the search process

    Parameters
    ----------
    search : str
        The abosolute path of `search` command.
    timeout_second : int
        The timeout second
    value_range : int
        The largest absolute value used in the dataset.
    examples : List[Example]
        The I/O examples used in the search process.
        This function find the program that matches the I/O examples.
    max_program_length : int
        The maximum length of the program
    pred : Function from List[Example] to Dict[str, float]
        The predict function. It receives the examples as inputs, and returns
        the probabilities of functions and lambdas.

    Returns
    -------
    SearchResult
        The result of the search procedure
    """
    # Use temporary directory to conduct search
    with tempfile.TemporaryDirectory() as tmpdir:
        name = os.path.join(tmpdir, "data", "search")
        os.makedirs(name)

        # Dump {input|output}_{types|values}.txt
        with open(os.path.join(name, "input_types.txt"), "w") as f:
            intypes = []
            for input in examples[0].inputs:
                inarr = np.array(input)
                if inarr.shape == ():
                    # Int
                    intypes.append("Int")
                else:
                    # IntList
                    intypes.append("Array")
            f.write(" ".join(intypes))
        with open(os.path.join(name, "input_values.txt"), "w") as f:
            values = []
            for example in examples:
                value = []
                for input in example.inputs:
                    inarr = np.array(input)
                    if inarr.shape == ():
                        value.append(str(input))
                    else:
                        value.append(" ".join(list(map(str, inarr))))
                values.append(" | ".join(value))
            f.write("\n".join(values))
        with open(os.path.join(name, "output_types.txt"), "w") as f:
            outtypes = []
            output = examples[0].output
            outarr = np.array(output)
            if outarr.shape == ():
                # Int
                outtypes.append("Int")
            else:
                # IntList
                outtypes.append("Array")
            f.write(" ".join(outtypes))
        with open(os.path.join(name, "output_values.txt"), "w") as f:
            values = []
            for example in examples:
                value = []
                outarr = np.array(example.output)
                if outarr.shape == ():
                    value.append(str(example.output))
                else:
                    value.append(" ".join(list(map(str, outarr))))
                values.append(" | ".join(value))
            f.write("\n".join(values))

        # Get probabilities
        try:
            prob = pred(examples)
        except:
            return SearchResult(False, dict([]), "", -1, -1)

        # Dump the probabilities to the file
        with open(os.path.join(name, "prior.txt"), "w") as f:
            probs = ["{} {}".format(p, name) for name, p in prob.items()]
            f.write("\n".join(probs))

        # Execute search command
        try:
            res = subprocess.run(
                [search, "search", str(len(examples)), str(
                    max_program_length), "0", "0", "-1", str(value_range)],
                stdout=subprocess.PIPE,
                timeout=timeout_second, cwd=tmpdir)
            lines = res.stdout.decode().split("\n")
        except subprocess.TimeoutExpired:
            return SearchResult(False, prob, "", -1, timeout_second)

        for i, line in enumerate(lines):
            if line == "Solved!":
                solution = "\n".join(lines[i + 4:])
                explored_nodes = int(
                    lines[i + 1].replace("Nodes explored: ", ""))
                time = float(lines[i + 2])
                return SearchResult(
                    True, prob, solution, explored_nodes, time)

        return SearchResult(False, prob, "", -1, -1)


def predict_with_prior_distribution(dataset):
    """
    Predict by using the prior distribution of the dataset

    Parameters
    ----------
    dataset : chainer.dataset
        The training dataset

    Returns
    -------
    function
        The predict function
    """
    prior = prior_distribution(dataset)
    return lambda x: prior


def predict_with_neural_network(model_shape: ModelShapeParameters, model: InferenceModel):
    """
    Predict by using the neural network

    Parameters
    ----------
    model_shape : ModelShapeParameters
        The parameters of the neural network model.
        It is used to interpret the output of the neural network.
    model : InferenceModel
        The deep neural network model.

    Returns
    -------
    function
        The predict function
    """
    def pred(examples: List[Example]):
        encodings = examples_encoding(examples, model_shape.dataset_metadata)
        pred = model.model(np.array([encodings.types]), np.array(
            [encodings.values])).array[0]
        retval = dict()
        for name, p in zip(sorted(list(model_shape.dataset_metadata.symbols)), pred):
            retval[name] = p
        return retval
    return pred
