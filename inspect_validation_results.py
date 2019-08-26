import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from typing import List, Tuple, Union, Dict, Callable, Set
from src.dataset import Dataset, prior_distribution
from src.inference import SearchResult

SEED_MAX = 2**32 - 1

parser = argparse.ArgumentParser()
parser.add_argument("--num-entries", help="The number of entries to dump", type=int, default=5)
parser.add_argument("--seed", help="The random seed", type=int, default=22956)
parser.add_argument("--width", help="The width of plots", type=int, default=16)
parser.add_argument("--height", help="The height of plots", type=int, default=6)
parser.add_argument("dataset", help="The validation dataset", type=str)
parser.add_argument("baseline", help="The baseline (use-prior) result", type=str)
parser.add_argument("result", help="The result with neural network", type=str)
args = parser.parse_args()

root_rng = np.random.RandomState(args.seed)
random.seed(root_rng.randint(SEED_MAX))
np.random.seed(root_rng.randint(SEED_MAX))

with open(args.dataset, "rb") as f:
    dataset: Dataset = pickle.load(f)
with open(args.baseline, "rb") as f:
    baseline: Dict[int, SearchResult] = pickle.load(f)
with open(args.result, "rb") as f:
    result: Dict[int, SearchResult] = pickle.load(f)

plt.ion()

# the time to solve
time_baseline = np.array([r.time_seconds for r in baseline.values() if r.is_solved])
time_baseline_include_timeout = np.array([r.time_seconds for r in baseline.values() if r.time_seconds > 0])
time_dnn = np.array([r.time_seconds for r in result.values() if r.is_solved])
time_dnn_include_timeout = np.array([r.time_seconds for r in result.values() if r.time_seconds > 0])

# the explored nodes
nodes_baseline = np.array([r.explored_nodes for r in baseline.values() if r.is_solved])
nodes_dnn = np.array([r.explored_nodes for r in result.values() if r.is_solved])

# Compare the result with baseline
fig, [ax_stats, ax_time] = plt.subplots(2, 1, figsize=(args.width, args.height))
## the number of solved entries
solved_baseline = len([r for r in baseline.values() if r.is_solved])
solved_dnn = len([r for r in result.values() if r.is_solved])
ax_stats.axis("tight")
ax_stats.axis("off")
ax_stats.set_title("Stats")
    
colLabels = ["Baseline", "DNN"]
rowLabels = ["# solved entries",
             "average time [second]", "median time [second]",
             "average time (including timeout) [second]", "median time (including timeout) [second]",
             "average # explored nodes", "median # explored nodes"]
text = [
    [solved_baseline, solved_dnn],
    [time_baseline.mean(), time_dnn.mean()],
    [np.median(time_baseline), np.median(time_dnn)],
    [time_baseline_include_timeout.mean(), time_dnn_include_timeout.mean()],
    [np.median(time_baseline_include_timeout), np.median(time_dnn_include_timeout)],
    [nodes_baseline.mean(), nodes_dnn.mean()],
    [np.median(nodes_baseline), np.median(nodes_dnn)]
]
ax_stats.table(cellText=text, colLabels=colLabels, rowLabels=rowLabels, loc="center")

## Plot the time to solve
xs_baseline = np.arange(solved_baseline) + 0.0
xs_dnn = np.arange(solved_dnn) + 0.2
ax_time.bar(xs_baseline, time_baseline, width=0.4, bottom=np.zeros(1), label="Baseline")
ax_time.bar(xs_dnn, time_dnn, width=0.4, bottom=np.zeros(1), label="DNN")
ax_time.legend()
ax_time.set_ylabel("Time [Second]")
ax_time.set_title("The Search Time")

## Show randomly chosen entries
indexes = np.random.choice(list(range(len(dataset.entries))), args.num_entries)
for index in indexes:
    entry = dataset.entries[index]
    baseline_result = baseline[index]
    dnn_result = result[index]

    fig, [ax_stats, ax_examples, ax_code] = plt.subplots(3, 1, figsize=(args.width, args.height))
    fig.suptitle("Entry {}".format(index))

    ax_stats.axis("tight")
    ax_stats.axis("off")
    ax_stats.set_title("Stats")

    colLabels = ["Baseline", "DNN"]
    rowLabels = ["time [second]", "# explored nodes"]
    text = [
        [baseline_result.time_seconds, dnn_result.time_seconds],
        [baseline_result.explored_nodes, dnn_result.explored_nodes]]
    ax_stats.table(cellText=text, colLabels=colLabels, rowLabels=rowLabels, loc="center")

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

    ax_code.axis("tight")
    ax_code.axis("off")
    ax_code.set_title("Source Code")
    colLabels = ["Ground Truth", "Baseline", "DNN"]
    text = [[entry.source_code, baseline_result.solution, dnn_result.solution]]
    ax_code.table(cellText=text, colLabels=colLabels, loc="center")

plt.show()
input("Press Enter to continue")
