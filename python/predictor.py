import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import sys

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

if len(sys.argv) == 3:
    # Predict attribute
    print("bar")
else:
    # Evaluate
    embed = deepCoder.encoder.embed.valueEmbed.integerEmbed
    for l in range(0, M.integer_range + 1):
        embedInteger = embed(np.array([l], dtype=np.float32)).data[0]
        s = str(l + M.integer_min)
        for x in embedInteger:
            s += " " + str(x)
        print(s)
