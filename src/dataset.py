import dataclasses
import numpy as np
import chainer as ch
from chainer import datasets
from typing import List, Union, Dict, Set
from .dsl import Function

Primitive = Union[int, List[int]]


@dataclasses.dataclass
class Example:
    """
    An I/O example

    Attributes
    ----------
    inputs : List[Primitive]
    output : Primitive
    """
    inputs: List[Primitive]
    output: Primitive


@dataclasses.dataclass
class Entry:
    """
    The entry of this dataset

    Attributes
    ----------
    source_code : str
        The source code of the program
    examples : list of Example
        The input/output examples for the source code
    attribute : dict from str to bool
        The binary attribute of the source code.
        The key represents the name of functions or lambdas (symbols).
        The value represents whether the program contains
        the function or not.
    """
    source_code: str
    examples: List[Example]
    attribute: Dict[str, bool]


@dataclasses.dataclass
class DatasetMetadata:
    max_num_inputs: int
    symbols: Set[str]
    value_range: int
    max_list_length: int


@dataclasses.dataclass
class Dataset:
    dataset: ch.datasets.TupleDataset
    metadata: DatasetMetadata

def dataset_metadata(dataset, value_range: int = -1, max_list_length: int = -1) -> DatasetMetadata: # TODO
    """
    Return the values for specifying the model shape

    Parameters
    ----------
    dataset : chainer.dataset
    value_range : int
        The largest absolute value used in the dataset
    max_list_length: int

    Returns
    -------
    DatasetMetadata
    """
    num_inputs = 0
    symbols = set([])
    for entry in dataset:
        entry = entry[0]
        num_inputs = max(num_inputs, len(entry.examples[0].inputs))
        if len(symbols) == 0:
            for symbol in entry.attribute.keys():
                symbols.add(symbol)
    return DatasetMetadata(num_inputs, symbols, value_range, max_list_length)


def prior_distribution(dataset) -> Dict[str, float]:
    """
    Return the prior distribution over functions

    Parameters
    ----------
    dataset : chainer.dataset
        The dataset to calculate the prior distribution.
        Each element of the dataset should be Tuple[Entry].

    Returns
    -------
    prior : Dict[str, float]
        The value represents the frequency of the function or lambda in the dataset.
    """

    prior: Dict[Function, float] = dict()
    for entry in dataset:
        entry = entry[0]
        for symbol, value in entry.attribute.items():
            if not symbol in prior:
                prior[symbol] = 0
            prior[symbol] += 1 if value else 0

    for symbol in prior.keys():
        prior[symbol] /= len(dataset)

    return prior


@dataclasses.dataclass
class PrimitiveEncoding:
    """
    A encoding of Primitive

    Attributes
    ----------
    t : int
        It represents the type of the primitive.
        0 means that the type is Int, and 1 means that the type is List[Int]
    value_arr : np.array
        The array of the values.
        The empty elements are filled with Null value.
    """
    t: int
    value_arr: np.array

@dataclasses.dataclass
class ExamplesEncoding:
    """
    A encoding of the list of Example

    Attributes
    ----------
    types : np.array
        The encoding of inputs and output types.
        The shape is (E, I + 1, 2) where
            E is the number of examples and
            I is the maximum number of inputs
    values: np.array
        The encoding of inputs and output values.
        The shape is (E, I + 1, max_list_length) where
            E is the number of examples,
            I is the maximum number of inputs, and
            max_list_length is the maximum length of the list.
    """
    types: np.array
    values: np.array

@dataclasses.dataclass
class EntryEncoding:
    """
    A encoding of Entry

    Attributes
    ----------
    examples: ExamplesEncoding
    attribute: np.array
    """
    examples: ExamplesEncoding
    attribute: np.array


def primitive_encoding(p: Primitive, metadata: DatasetMetadata) -> PrimitiveEncoding:
    """
    Parameters
    ----------
    p : Primitive
        The primitive to encode
    metadata : DatasetMetadata

    Returns
    -------
    PrimitiveEncoding
        The encoding of the primitive
    """
    Null = metadata.value_range
    arr = np.array(p)

    t = 0 if arr.shape == () else 1
    value_arr = np.ones((metadata.max_list_length,)) * Null
    value_arr[:arr.size] = arr

    # Add offset of value_range because the range of integers is [-value_range:value_range-1]
    return PrimitiveEncoding(t, value_arr + metadata.value_range)


def examples_encoding(examples: List[Example], metadata: DatasetMetadata) -> ExamplesEncoding:
    E = len(examples)
    I = metadata.max_num_inputs
    max_list_length = metadata.max_list_length
    Null = metadata.value_range * 2

    types = np.zeros((E, I + 1, 2), dtype=np.int32)
    values = np.ones((E, I + 1, max_list_length), dtype=np.int32) * Null
    for i, example in enumerate(examples):
        if len(example.inputs) > I:
            raise RuntimeError("The number of inputs ({}) exceeds the limits ({})".format(
                len(example.inputs), I))
        enc_inputs = [primitive_encoding(ins, metadata) for ins in example.inputs]
        enc_output = primitive_encoding(example.output, metadata)
        types[i, :len(example.inputs), :] = [np.identity(2)[enc_input.t] for enc_input in enc_inputs]
        values[i, :len(example.inputs), :] = [enc_input.value_arr for enc_input in enc_inputs]
        types[i, I, :] = np.identity(2)[enc_output.t]
        values[i, I, :] = enc_output.value_arr

    return ExamplesEncoding(types, values)

def attribute_encoding(attribute: Dict[Function, bool]) -> np.array:
    """
    Parameters
    ----------
    attribute : Dict[Function, bool]
        The binary attribute

    Returns
    -------
    PrimitiveEntry
        The encoding of the entry
    """
    symbols = list(attribute.keys())
    symbols = sorted(symbols)
    arr = []
    for symbol in symbols:
        arr.append(1 if attribute[symbol] else 0)

    return np.array(arr, dtype=np.int32)


def entry_encoding(entry: Entry, metadata: DatasetMetadata) -> EntryEncoding:
    examples = examples_encoding(entry.examples, metadata)
    attribute = attribute_encoding(entry.attribute)
    return EntryEncoding(examples, attribute)


class EncodedDataset(datasets.TransformDataset):
    """
    The dataset of the entry encodings for DeepCoder
    This instance stores each entry as the tuple of
    (the encoding of examples, the encoding of attribute).
    """

    def __init__(self, dataset: Dataset):
        """
        Constructor

        Parameters
        ----------
        dataset : Dataset
            The dataset and its metadata
        """

        def transform(in_data):
            entry = in_data[0]
            encoding = entry_encoding(entry, dataset.metadata)
            return encoding.examples, encoding.attribute

        super(EncodedDataset, self).__init__(dataset.dataset, transform)
