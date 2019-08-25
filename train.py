import numpy as np
import random
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import chainer as ch
from chainer import datasets
from chainer.training import extensions
from src.chainer_dataset import ChainerDataset
from src.dataset import Dataset
import src.train as T

SEED_MAX = 2**32 - 1

parser = argparse.ArgumentParser()
parser.add_argument("--single-batch", help="Use one entry to debug the neural network model", action="store_true")
parser.add_argument("--value-range", help="The largest absolute value used in the dataset", type=int, default=256)
parser.add_argument("--max-list-length", help="The maximum length of the list used in the dataset", type=int, default=20)
parser.add_argument("--num-epochs", help="The number of epoch", type=int, default=50)
parser.add_argument("--seed", help="The random seed", type=int, default=24649)
parser.add_argument("--n-embed", help="The dimension of integer embeddings", type=int, default=20)
parser.add_argument("--n-units", help="The number of units in the hidden layers", type=int, default=256)
parser.add_argument("--batch-size", help="The minibatch-size", type=int, default=32)
parser.add_argument("--ratio-test", help="The ratio of entries for evaluation.", type=float, default=None)
parser.add_argument("--device", help="The device used for training.", type=int, default=-1)
parser.add_argument("dataset", help="The path of the dataset pickle file", type=str)
parser.add_argument("output", help="The directory to store the output", type=str)
args = parser.parse_args()

root_rng = np.random.RandomState(args.seed)
random.seed(root_rng.randint(SEED_MAX))
np.random.seed(root_rng.randint(SEED_MAX))

with open(args.dataset, "rb") as f:
    dataset: Dataset = pickle.load(f)

if args.single_batch:
    dataset = Dataset([dataset.entries[root_rng.randint(len(dataset.entries))]])

dataset_stats = T.dataset_stats(dataset)
model_shape = T.ModelShapeParameters(dataset_stats, args.value_range, args.max_list_length, args.n_embed, args.n_units)
model = T.model(model_shape)

# Save model shape
if not os.path.exists(args.output):
    os.makedirs(args.output)
assert(os.path.isdir(args.output))
with open(os.path.join(args.output, "model-shape.pickle"), "wb") as f:
    pickle.dump(model_shape, f)

n_entries = len(dataset.entries)
dataset = ChainerDataset(dataset, args.value_range, args.max_list_length)
if args.ratio_test is None or args.ratio_test == 0:
    train = dataset
    test = None
else:
    train, test = datasets.split_dataset_random(dataset, int(n_entries * (1.0 - args.ratio_test)), seed=root_rng.randint(SEED_MAX))

train_iter = ch.iterators.SerialIterator(train, args.batch_size)
if test is not None:
    test_iter = ch.iterators.SerialIterator(test, args.batch_size, repeat=False, shuffle=False)
else:
    test_iter = None

trainer = T.trainer(train_iter, args.output, model, args.num_epochs, device=args.device)
if test_iter is not None:
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.device))
trainer.extend(extensions.LogReport())
if test_iter is not None:
    trainer.extend(extensions.PrintReport(
                ['epoch', 'main/loss', 'validation/main/loss',
                 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
else:
    trainer.extend(extensions.PrintReport(
                ['epoch', 'main/loss', 'main/accuracy', 'elapsed_time']))
trainer.extend(extensions.snapshot(filename="snapshot_{.updater.epoch}"), trigger=(10, 'epoch'))
trainer.extend(extensions.snapshot_object(model, "model_{.updater.epoch}"), trigger=(10, 'epoch'))
trainer.run()
