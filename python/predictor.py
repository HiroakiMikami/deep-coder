import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import model

embed = model.ExampleEmbed(1, 2, 2)
encoder = model.Encoder(embed, 10)
decoder = model.Decoder(33)
model = model.DeepCoder(encoder, decoder)

#serializers.load_npz("test.dat", model)

x = np.array([[[1, 0, 1, 2], [0, 1, 1, 1]], [[1, 0, 1, 2], [0, 0, 2, 2]], [[1, 0, 1, 2], [0, 0, 2, 2]]], dtype=np.float32)
print(x.shape)
#print(model(x))