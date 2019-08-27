import dataclasses
import numpy as np
from typing import List, Union, Tuple, Dict
from .dsl import Function

Primitive = Union[int, List[int]]
Example = Tuple[List[Primitive], Primitive]


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
    attributes : dict from str to bool
        The binary attributes of the source code.
        The key represents the name of functions or lambdas.
        The value represents whether the program contains
        the function or not.
    """
    source_code: str
    examples: List[Example]
    attributes: Dict[str, bool]


@dataclasses.dataclass
class Dataset:
    """
    The dataset

    Attributes
    ----------
    entries : list of Entry
        The entries of this dataset.
    """
    entries: List[Entry]


def prior_distribution(dataset: Dataset) -> Dict[str, float]:
    """
    Return the prior distribution over functions

    Parameters
    ----------
    dataset : Dataset
        The dataset to calculate the prior distribution

    Returns
    -------
    prior : Dict[str, float]
        The value represents the frequency of the function or lambda in the dataset.
    """

    prior: Dict[Function, float] = dict()
    for entry in dataset.entries:
        for function, value in entry.attributes.items():
            if not function in prior:
                prior[function] = 0
            prior[function] += 1 if value else 0

    for function in prior.keys():
        prior[function] /= len(dataset.entries)

    return prior


def divide(dataset: Dataset, separators: Dict[str, int], rng: Union[None, np.random.RandomState] = None) -> Dict[str, Dataset]:
    """
    Divide the dataset into some sub-datasets and return the set of sub-datasets

    Attributes
    ----------
    dataset : Dataset
        The dataset to be divided
    separators : Dict[str, int]
        The dictionary used to specify how to divide the dataset.
        The keys represent the name of the sub-datasets, and the values represent
        the number of entries contained in the sub-datasets.
    rng : None or np.random.RandomState
        The generator of random numbers.

    Returns
    -------
    sub_datasets: Dict[str, Dataset]
        The set of sub-datasets.
    """

    if rng is None:
        rng = np.random

    total = sum(map(lambda x: x[1], separators.items()))
    assert(total <= len(dataset.entries))

    random_indexes = np.random.choice(
        len(dataset.entries), len(dataset.entries), replace=False)
    offset = 0
    retval = dict()
    for name, num in separators.items():
        d = Dataset([])
        for index in range(offset, offset + num):
            d.entries.append(dataset.entries[random_indexes[index]])
        retval[name] = d
        offset += num

    return retval
