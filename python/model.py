import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class IntegerEmbed(Chain):
    def __init__(self, integer_range, embed_length):
        super(IntegerEmbed, self).__init__()
        with self.init_scope():
            self.id = L.EmbedID(integer_range + 1, embed_length)
    def __call__(self, x):
        # Input : [ID ([0, ..., integer_range-1, NULL(integer_range)])]
        # Output: [[float]^embed_length]
        return self.id(F.cast(x, np.int32))

class ValueEmbed(Chain):
    def __init__(self, integer_range, embed_length):
        super(ValueEmbed, self).__init__()
        with self.init_scope():
            self.integerEmbed = IntegerEmbed(integer_range, embed_length)
    def __call__(self, x):
        # Input: [Value([type-vector (valueID)*])]
        # Output: [[double+]]
        (n1, n2) = x.shape
        (t, l) = F.split_axis(x, [2], 1)
        embedL_ = self.integerEmbed(l)
        embedL = embedL_.reshape([n1, int(embedL_.size / n1)])
        return F.concat((t, embedL))

class ExampleEmbed(Chain):
    def __init__(self, input_num, integer_range, embed_length):
        super(ExampleEmbed, self).__init__()
        with self.init_scope():
            self.valueEmbed = ValueEmbed(integer_range, embed_length)
    def __call__(self, x):
        # Input: [Example([Value])]
        # Output: [[double+]]
        (n1, n2, n3) = x.shape
        x1 = x.reshape(n1 * n2, n3)
        x_ = self.valueEmbed(x1)
        return x_.reshape([n1, int(x_.size / n1)])

class Encoder(Chain):
    def __init__(self, embed, out_width):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.embed = embed
            self.h1 = L.Linear(None, out_width)
            self.h2 = L.Linear(None, out_width)
            self.h3 = L.Linear(None, out_width)
    def __call__(self, x):
        # Input: [Example]
        # Output: [[double+]]
        e = self.embed(x)

        x1 = F.sigmoid(self.h1(e))
        x2 = F.sigmoid(self.h2(x1))
        x3 = F.sigmoid(self.h2(x2))
        return x3

class Decoder(Chain):
    def __init__(self, out_width):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.h = L.Linear(None, out_width)
    def __call__(self, x):
        # Input: [[[double+]]]
        # Output: [[double+](attribute before sigmoid)]
        x1 = F.average(x, axis=1)
        x2 = self.h(x1)
        return x2

class DeepCoder(Chain):
    def __init__(self, encoder, decoder, example_num):
        super(DeepCoder, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder
            self.example_num = example_num
    def __call__(self, x):
        # Input: [[Example]]
        # Output: [[double]+(attribute before sigmoid)]
        (data_num, example_num, n3, n4) = x.shape
        x1 = self.encoder(x.reshape(data_num * example_num, n3, n4))
        (_, vec_length) = x1.shape
        x2 = self.decoder(x1.reshape(data_num, example_num, vec_length))
        return x2

input_num = 3
embed_length =20 
integer_min = -100
integer_max = 100
integer_range = integer_max - integer_min + 1 #integerの個数
example_num = 5
list_length = 10
hidden_layer_width = 256 
attribute_width = 34

def gen_model():
    embed = ExampleEmbed(input_num, integer_range, embed_length)
    encoder = Encoder(embed, hidden_layer_width)
    decoder = Decoder(attribute_width)
    deepCoder = DeepCoder(encoder, decoder, example_num)
    return deepCoder

def convert_integer(integer):
    return integer -integer_min # integer_min -> 0
def type_vector(value):
    if isinstance(value, list):
        return [0, 1]
    elif isinstance(value, int):
        return [1, 0]
    else:
        return [0, 0]
def convert_value(value):
    # type vector
    t = type_vector(value)
    if isinstance(value, int):
        value = [value]
    elif isinstance(value, list):
        value = value
    else:
        value = []
    value = [convert_integer(x) for x in value]
    # Fill NULL (integer_range)
    if len(value) < list_length:
        add = [integer_range] * (list_length - len(value))
        value.extend(add)
    t.extend(value)
    return np.array(t, dtype=np.float32)

def convert_example(example):
    # Fill NULL input
    input = example['input']
    if len(input) < input_num:
        add = [""] * (input_num - len(input))
        input.extend(add)
    output = example['output']
    x = [convert_value(y) for y in input]
    x.extend([convert_value(output)])
    return np.array(x)

def convert_each_data(data):
    examples = data['examples']
    # Convert
    examples2 = np.array([convert_example(x) for x in examples])
    attrs = np.array(data['attribute'], dtype=np.int32)
    return examples2, attrs

def preprocess_json(data):
    return [convert_each_data(x) for x in data]
