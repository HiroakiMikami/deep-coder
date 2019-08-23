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
