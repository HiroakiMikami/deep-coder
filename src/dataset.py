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
class ExampleEncoding:
    """
    A encoding of Example

    Attributes
    ----------
    inputs : List[PrimitiveEncoding]
    output : PrimitiveEncoding
    """
    inputs: List[PrimitiveEncoding]
    output: PrimitiveEncoding


@dataclasses.dataclass
class EntryEncoding:
    """
    A encoding of Entry

    Attributes
    ----------
    examples: List[ExampleEncoding]
    attribute: np.array
    """
    examples: List[ExampleEncoding]
    attribute: np.array


def encode_primitive(p: Primitive, metadata: DatasetMetadata) -> PrimitiveEncoding:
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


def encode_example(example: Example, metadata: DatasetMetadata) -> ExampleEncoding:
    enc_inputs = [encode_primitive(ins, metadata) for ins in example.inputs]
    enc_output = encode_primitive(example.output, metadata)
    return ExampleEncoding(enc_inputs, enc_output)


def encode_attribute(attribute: Dict[Function, bool]) -> np.array:
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

    return np.array(arr)


def encode_entry(entry: Entry, metadata: DatasetMetadata) -> EntryEncoding:
    example_encoding = [encode_example(example, metadata)
                        for example in entry.examples]
    example_attribute = encode_attribute(entry.attribute)
    return EntryEncoding(example_encoding, example_attribute)


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
            encoding = encode_entry(entry, dataset.metadata)
            return encoding.examples, encoding.attribute

        super(EncodedDataset, self).__init__(dataset.dataset, transform)
