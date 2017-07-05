import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import json
import sys
import traceback
import model as M

class Model(Chain):
    def __init__(self, deepCoder):
        super(Model, self).__init__()
        with self.init_scope():
            self.deepCoder = deepCoder
    def __call__(self, input_, output):
        actual = self.deepCoder(Variable(input_))
        loss = F.sigmoid_cross_entropy(actual, Variable(output))
        report({'loss': loss}, self)
        return loss

def preprocess_json(data):
    def convert_integer(integer):
        return integer - M.integer_min # integer_min -> 0
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
        if len(value) < M.list_length:
            add = [M.integer_range] * (M.list_length - len(value))
            value.extend(add)
        t.extend(value)

        return np.array(t, dtype=np.float32)
    def convert_example(example):
        # Fill NULL input
        input = example['input']
        if len(input) < M.input_num:
            add = [""] * (M.input_num - len(input))
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
    return [convert_each_data(x) for x in data]

class Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, data):
        super(chainer.dataset.DatasetMixin, self).__init__()
        self.data = data
    def __len__(self):
        return len(self.data)
    def get_example(self, i):
        return self.data[i]

deepCoder = M.gen_model()
model = Model(deepCoder)

f = open(sys.argv[1], 'r')
x = json.load(f) # [Example]
y = preprocess_json(x) #[Example(postprocessed)]

l1 = [e for e in range(0, len(y)) if e % 6 != 0]
l2 = [e for e in range(0, len(y)) if e % 6 == 0]

train = Dataset([y[e] for e in range(0, len(y))])
test = Dataset([y[e] for e in l2])

try:
    train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
    test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

    deepCoder = M.gen_model()
    model = Model(deepCoder)
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (20, 'epoch'), out='result')

    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    serializers.save_npz(sys.argv[2], deepCoder)
except:
    print(traceback.format_exc())

