import dataclasses
import numpy as np
from typing import List, Union, Dict, Set
import chainer as ch
from chainer import training
from chainer.training import extensions
from chainer import cuda
from .model import ModelShapeParameters, Predictor, TrainingClassifier


def convert_entry(batch, device):
    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        def to_device(x):
            return cuda.to_gpu(x, device, cuda.Stream.null)

    return (to_device(np.array([types for types, _, _ in batch])),
            to_device(np.array([values for _, values, _ in batch])),
            np.array([attribute for _, _, attribute in batch]))


class Training:
    """
    Store the instances for training

    Attributes
    ----------
    predictor : ch.Link
        The attribute predictor of DeepCoder
    model : ch.Link
    trainer : training.Trainer
    """

    def __init__(self,
                 train_iter, test_iter, out: str,
                 params: ModelShapeParameters, w_0: float,
                 num_epochs: int, optimizer=ch.optimizers.Adam(), device=-1):
        """
        Constructor

        Parameters
        ----------
        train_iter : iterator
            The iterator of the training dataset
        test_iter : iterator or None
            The iterator of the est dataset
        out : str
            The path of the output directory
        params : ModelShapeParames
        w_0 : float
        The weight for label=False
        num_epochs : int
            The number of epochs
        optimizer
        device : int
            The device used for training
        """

        self.predictor = Predictor(params)
        self.model = TrainingClassifier(self.predictor, w_0)
        opt = optimizer.setup(self.model)
        updater = training.StandardUpdater(
            train_iter, optimizer, device=device, converter=convert_entry)
        self.trainer = training.Trainer(
            updater, (num_epochs, "epoch"), out=out)
        if test_iter is not None:
            self.trainer.extend(extensions.Evaluator(
                test_iter, self.model, device=device, converter=convert_entry))
