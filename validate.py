import numpy as np
import random
import pickle
import os
import argparse
from tqdm import tqdm
import chainer as ch
import src.inference as I
from src.dataset import Dataset
from src.model import ModelShapeParameters

SEED_MAX = 2**32 - 1

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use-prior", help="Use prior distribution in prediction", type=str, default=None)
parser.add_argument("--max-program-length",
                    help="The maximum length of the program", type=int, default=5)
parser.add_argument("--timeout-second", help="The timeout",
                    type=int, default=10)
parser.add_argument("--seed", help="The random seed", type=int, default=24649)
parser.add_argument(
    "modelshape", help="The path of the model shape parameter file", type=str)
parser.add_argument("model", help="The path of the model file", type=str)
parser.add_argument(
    "dataset", help="The path of the dataset pickle file", type=str)
parser.add_argument(
    "output", help="The path to store the result", type=str)
args = parser.parse_args()

root_rng = np.random.RandomState(args.seed)
random.seed(root_rng.randint(SEED_MAX))
np.random.seed(root_rng.randint(SEED_MAX))

if not os.path.exists(os.path.dirname(args.output)):
    os.makedirs(os.path.dirname(args.output))

with open(args.dataset, "rb") as f:
    dataset: Dataset = pickle.load(f)

# Load model
with open(args.modelshape, "rb") as f:
    model_shape: ModelShapeParameters = pickle.load(f)
model = I.InferenceModel(model_shape)
ch.serializers.load_npz(args.model, model.predictor)

assert(model_shape.dataset_metadata == dataset.metadata)

if args.use_prior:
    with open(args.use_prior, "rb") as f:
        train_dataset: ch.datasets.TupleDataset = pickle.load(f).dataset
    pred = I.predict_with_prior_distribution(train_dataset)
else:
    pred = I.predict_with_neural_network(model_shape, model)

results = dict([])
num_succ = 0
for i, (entry,) in enumerate(tqdm(dataset.dataset)):
    result = I.search(
        os.path.join(os.getcwd(), "DeepCoder_Utils",
                     "enumerative-search", "search"),
        args.timeout_second,
        model_shape.dataset_metadata.value_range,
        entry.examples,
        args.max_program_length,
        pred
    )
    results[i] = result
    if result.is_solved:
        num_succ += 1

print("Solved: {} of {} examples".format(num_succ, len(dataset.dataset)))

with open(args.output, "wb") as f:
    pickle.dump(results, f)
