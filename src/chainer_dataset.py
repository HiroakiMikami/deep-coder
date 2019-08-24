import dataclasses
from chainer import datasets
import numpy as np
from typing import List, Union, Tuple, Dict

from .dataset import Dataset, Primitive, Example
from .dsl import Function

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

def encode_primitive(p: Primitive, value_range: int, max_list_length: int) -> PrimitiveEncoding:
    """
    Parameters
    ----------
    p : Primitive
        The primitive to encode
    value_range : int
        The largest absolute value used in the dataset
    max_list_length : int
        The maximum length of the list used in the dataset

    Returns
    -------
    PrimitiveEncoding
        The encoding of the primitive
    """
    Null = value_range
    arr = np.array(p)

    t = 0 if arr.shape == () else 1
    value_arr = np.ones((max_list_length,)) * Null
    value_arr[:arr.size] = arr

    # Add offset of value_range because the range of integers is [-value_range:value_range-1]
    return PrimitiveEncoding(t, value_arr + value_range)

def encode_example(example: Example, value_range: int, max_list_length: int) -> ExampleEncoding:
    inputs, output = example
    enc_inputs = [encode_primitive(ins, value_range, max_list_length) for ins in inputs]
    enc_output = encode_primitive(output, value_range, max_list_length)
    return ExampleEncoding(enc_inputs, enc_output)

def encode_attribute(attribute: Dict[Function, bool]) -> np.array:
    """
    Parameters
    ----------
    attribute : Dict[Function, bool]
        The binary attribute
    value_range : int
        The largest absolute value used in the dataset
    max_list_length : int
        The maximum length of the list used in the dataset

    Returns
    -------
    PrimitiveEntry
        The encoding of the entry
    """
    func_names = list(attribute.keys())
    func_names = sorted(func_names, key=lambda x: x.name)
    arr = []
    for func in func_names:
        arr.append(1 if attribute[func] else 0)

    return np.array(arr)

class ChainerDataset(datasets.TupleDataset):
    """
    The dataset of chainer for DeepCoder
    This instance stores each entry as the tuple of
    (the encoding of examples, the encoding of attribute).
    """
    def __init__(self, dataset: Dataset, value_range: int, max_list_length: int):
        """
        Constructor

        Parameters
        ----------
        dataset : Dataset
            The instance contains the entries of the program, examples, and attributes
        value_range : int
            The largest absolute value used in the dataset
        max_list_length : int
            The maximum length of the list used in the dataset
        """

        examples = []
        attributes = []
        for entry in dataset.entries:
            E = [encode_example(example, value_range, max_list_length) for example in entry.examples]
            A = encode_attribute(entry.attributes)
            examples.append(E)
            attributes.append(A)
        super(ChainerDataset, self).__init__(examples, attributes)
