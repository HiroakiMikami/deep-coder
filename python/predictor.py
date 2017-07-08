import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import sys
import json
import copy

import model as M

class Predictor(Chain):
    def __init__(self, deepCoder):
        super(Predictor, self).__init__()
        with self.init_scope():
            self.deepCoder = deepCoder

    def __call__(self, x):
        return F.sigmoid(self.deepCoder(x))
deepCoder = M.gen_model()
serializers.load_npz(sys.argv[1], deepCoder)

predictor = Predictor(deepCoder)

def print_value(value):
    s = ''
    if isinstance(value, int):
        s = "Integer " + str(value)
    else:
        s = "List "
        for x in value:
            s += str(x) + " "
    print(s)

if len(sys.argv) >= 3:
    # Predict attribute
    f = open(sys.argv[2], 'r')
    examples_ = json.load(f) # [Example]
    e = copy.deepcopy(examples_)
    examples = np.array([M.convert_example(x) for x in examples_])

    # Parse Examples
    for example in e:
        i = example["input"]
        o = example["output"]

        for value in i:
            print_value(value)
        print("---")
        print_value(o)
        print("---")

    print("---")
    # Output Attributes
    if len(sys.argv) >= 4 and sys.argv[3] == "none":
        x = 'Attribute: '
        for t in range(0, M.attribute_width):
            x += "1 "
        print(x)
    else:
        x = 'Attribute: '
        attributes = predictor(np.array([examples]))[0].data
        for t in attributes:
            x += str(t) + " "
        print(x)
else:
    # Evaluate
    embed = deepCoder.encoder.embed.valueEmbed.integerEmbed
    for l in range(0, M.integer_range + 1):
        embedInteger = embed(np.array([l], dtype=np.float32)).data[0]
        s = str(l + M.integer_min)
        for x in embedInteger:
            s += " " + str(x)
        print(s)
