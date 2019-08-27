import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import chainer as ch
import argparse
from src.dataset import Dataset
from src.chainer_dataset import ChainerDataset
import src.train as T

SEED_MAX = 2**32 - 1

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="The random seed", type=int, default=22956)
parser.add_argument("--width", help="The width of plots", type=int, default=16)
parser.add_argument("--height", help="The height of plots",
                    type=int, default=6)
parser.add_argument(
    "modelshape", help="The path of the model shape parameter file", type=str)
parser.add_argument("model", help="The path of the model file", type=str)
args = parser.parse_args()

root_rng = np.random.RandomState(args.seed)
random.seed(root_rng.randint(SEED_MAX))
np.random.seed(root_rng.randint(SEED_MAX))

# Load model
with open(args.modelshape, "rb") as f:
    model_shape: T.ModelShapeParameters = pickle.load(f)
model = T.model(model_shape)
ch.serializers.load_npz(args.model, model.predictor)

plt.ion()

# integer embeddings
fig, ax = plt.subplots(figsize=(args.width, args.height))
embed = list(model.predictor.children())[0]._embed_integer

axis_0, axis_1 = np.random.choice(model_shape.n_embed, 2, replace=False)
for i in range(-model_shape.value_range, model_shape.value_range):
    e = embed(np.array([i + model_shape.value_range]))
    x = e.array[0, axis_0]
    y = e.array[0, axis_1]

    if i == 0:
        color = "b"
    elif i > 0:
        color = "g"
    else:
        color = "r"

    if i % 2 == 0:
        shape = "s"
    else:
        shape = "^"

    ax.plot(x, y, "{}{}".format(color, shape))
    if abs(i) < 10 or abs(i) > 253:
        ax.annotate("{}".format(i), xy=(x, y))

e = embed(np.array([2 * model_shape.value_range]))
x = e.array[0, axis_0]
y = e.array[0, axis_1]
ax.plot(x, y, "x")
ax.annotate("Null", xy=(x, y))

plt.show()
input("Press Enter to continue")
