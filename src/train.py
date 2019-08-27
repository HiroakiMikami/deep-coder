import dataclasses
from typing import List, Union, Dict, Set
import chainer as ch
from chainer import training
from .model import ModelShapeParameters, Predictor, TrainingClassifier


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
                 train_iter, out: str,
                 params: ModelShapeParameters, w_0: float,
                 num_epochs: int, optimizer=ch.optimizers.Adam(), device=-1):
        """
        Constructor

        Parameters
        ----------
        train_iter : iterator
            The iterator of the training dataset
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
            train_iter, optimizer, device=device)
        self.trainer = training.Trainer(
            updater, (num_epochs, "epoch"), out=out)
