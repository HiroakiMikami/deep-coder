import dataclasses
from typing import List, Union, Tuple, Dict
import chainer as ch
from chainer import training
from .dataset import Dataset
from .model import ExampleEmbed, Encoder, Decoder, TrainingClassifier

@dataclasses.dataclass
class DatasetStats:
    max_num_inputs: int
    num_functions: int

@dataclasses.dataclass
class ModelShapeParameters:
    dataset_stats: DatasetStats
    value_range: int
    max_list_length: int
    n_embed: int
    n_units: int

def dataset_stats(dataset: Dataset) -> DatasetStats:
    """
    Return the values for specifying the model shape

    Parameters
    ----------
    dataset : Dataset

    Returns
    -------
    DatasetStats
        The maximum number of inputs and the number of functions
        in the dataset.
    """
    num_inputs = 0
    num_functions = 0
    for entry in dataset.entries:
        num_inputs = max(num_inputs, len(entry.examples[0][0]))
        if num_functions == 0:
            num_functions = len(entry.attributes)
    return DatasetStats(num_inputs, num_functions)

def model(params: ModelShapeParameters) -> ch.Link:
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
    embed = ExampleEmbed(params.dataset_stats.max_num_inputs, params.value_range, params.n_embed)
    encoder = Encoder(params.n_units)
    decoder = Decoder(params.dataset_stats.num_functions)
    return TrainingClassifier(embed, encoder, decoder)

def trainer(train_iter, out: str,
            model: ch.Link, num_epochs: int, optimizer = ch.optimizers.Adam(), device=-1):
    """
    Return the trainer

    Parameters
    ----------
    train_iter : iterator
        The iterator of the training dataset
    out : str
        The path of the output directory
    model : ch.Link
    num_epochs : int
        The number of epochs
    optimizer
    device : int
        The device used for training

    Returns
    -------
    ch.Link
        The model of DeepCoder
    """
    opt = optimizer.setup(model)
    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (num_epochs, "epoch"), out=out)
    return trainer
