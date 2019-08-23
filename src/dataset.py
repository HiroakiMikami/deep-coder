import dataclasses
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
        The value represents whether the program contains
        the function or not.
    """
    source_code: str
    examples: List[Example]
    attributes: Dict[Function, bool]

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

def prior_distribution(dataset: Dataset) -> Dict[Function, float]:
    """
    Return the prior distribution over functions

    Parameters
    ----------
    dataset : Dataset
        The dataset to calculate the prior distribution

    Returns
    -------
    prior : Dict[Function, float]
        The value represents the frequency of the function in the dataset.
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
