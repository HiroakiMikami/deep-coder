import numpy as np
import random
import pickle
from matplotlib import colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from src.dataset import Dataset, prior_distribution

SEED_MAX = 2**32 - 1

parser = argparse.ArgumentParser()
parser.add_argument("--num-entries", help="The number of entries to dump", type=int, default=5)
parser.add_argument("--seed", help="The random seed", type=int, default=22956)
parser.add_argument("--width", help="The width of plots", type=int, default=16)
parser.add_argument("--height", help="The height of plots", type=int, default=6)
parser.add_argument("dataset", help="The path of the dataset pickle file", type=str)
args = parser.parse_args()

root_rng = np.random.RandomState(args.seed)
random.seed(root_rng.randint(SEED_MAX))
np.random.seed(root_rng.randint(SEED_MAX))

with open(args.dataset, "rb") as f:
    dataset: Dataset = pickle.load(f)

plt.ion()

# prior-distribution
prior = prior_distribution(dataset)
columns = []
data = []
for name, prob in prior.items():
    columns.append(name)
    data.append(prob)
data = np.array([data])
data = pd.DataFrame(data, columns=columns)

## Show plot
fig, ax = plt.subplots(figsize=(args.width, args.height))
xs = np.arange(len(columns)) + 10
ax.bar(xs, data.iloc[0], width=0.4, bottom=np.zeros(1), tick_label=list(map(lambda x: x.replace(" ", "\n"), columns)))
ax.set_ylabel("Probability")
ax.set_title("Prior Distribution")

## Show randomly chosen entries
m = cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap=cm.Greens)
indexes = np.random.choice(list(range(len(dataset.entries))), args.num_entries)
for index in indexes:
    entry = dataset.entries[index]

    fig, [ax_code, ax_examples, ax_attributes] = plt.subplots(3, 1, figsize=(args.width, args.height))
    fig.suptitle("Entry {}".format(index))

    ax_code.axis("tight")
    ax_code.axis("off")
    ax_code.set_title("Source Code")
    ax_code.text(0, 0.5, entry.source_code)

    ax_examples.axis("tight")
    ax_examples.axis("off")
    ax_examples.set_title("Examples")
    num_inputs = max(map(lambda x: len(x[0]), entry.examples))
    colLabels = []
    for i in range(num_inputs):
        colLabels.append("Input {}".format(i + 1))
    colLabels.append("Output")
    rowLabels = []
    text = []
    for i, (ins, out) in enumerate(entry.examples):
        rowLabels.append("Example {}".format(i))
        row = []
        for i in ins:
            row.append(i)
        for i in range(len(ins), num_inputs):
            row.append("")
        row.append(out)
        text.append(row)
    ax_examples.table(cellText=text, colLabels=colLabels, rowLabels=rowLabels, loc="center")

    ax_attributes.set_title("Attributes")
    ax_attributes.get_yaxis().set_visible(False)
    data = np.ones(len(entry.attributes))
    colors = []
    for name, v in entry.attributes.items():
        colors.append(m.to_rgba(1 if v else 0))
    xs = np.arange(len(entry.attributes)) + 10
    ax_attributes.bar(xs, data, width=0.9, bottom=np.zeros(1),
            color=colors,
            tick_label=list(entry.attributes.keys()))

plt.show()
input("Press Enter to continue")
