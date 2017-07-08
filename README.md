deep-coder
===
> Re-implement DeepCoder (https://openreview.net/pdf?id=ByldLrqlx)


Requirements
---
* bash
* cmake (>= 3.0)
* g++ (>= 6.x.x)
* gtest
* python3
* chainer, numpy, mock (pip packages)

Build program
---
```bash
$ git clone https://github.com/HiroakiMikami/deep-coder.git
$ cd deep-coder
$ mkdir -p build
$ cd build
$ cmake ..
$ make
$ cd ../
```

1. Use Pre-Trained Model
---
```bash
# Search program using a newral network model
$ ./scripts/gen_program.sh model/model.dat examples/dot.json 2
 # The probability of each functions

 # Program
$ ./scripts/gen_program.sh model/model.dat examples/dot.json 2 none # Search program without a neural network model
# it will take very long time (I don't know how much time it will take).
```

2. Simple Experiment
---
The following commands do:
1. generate a small dataset (N=10000),
2. learn the neural network model by using the dataset, and
3. search the program that receives an list and outputs the reversed list.

```bash
# Generate dataset
$ ./build/src/gen_dataset 1 10000 300 > small_dataset.json
# Learn attributes
$ python3 ./python/learner.py small_dataset.json model.dat
# Search program using a newral network model
$ ./scripts/gen_program.sh model.dat examples/reverse.json 1 2> /dev/null
---
a <- read_list
b <- reverse a
---
```

3. Experiment with Larger Dataset
---

### a. Dataset generation
Generate a dataset (N=about 50000):
```bash
$ ./build/src/gen_dataset 3 50000 5 > dataset.json
```

NOTES: *It took over 10 hours on my laptop computer.*

#### Dataset Format
```json
[
    {
        "examples": [
            {
                "inputs": [[10, 1], 1],
                "output": 1
            },
            // ... input and output examples
            {
                "inputs": [[1, 2, 3, 4, 5], 3],
                "output": 4
            }
        ],
        "attribute": [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    },
    // ...examples...
    {
        "examples": [
            {...},
            ...
            {...}
        ],
        "attribute": ...
    }
]
```

### b. Learn attributes
```bash
$ python3 ./python/learner.py dataset.json model.dat
```

### c. Program search
```bash
$ ./scripts/gen_program.sh model.dat examples/reverse.json 1 2> /dev/null
---
a <- read_list
b <- reverse a
---
```

#### Example Format
```json
[
    {
        "inputs": [[10, 1], 1],
        "output": 1
    },
    // ... input and output examples
    {
        "inputs": [[1, 2, 3, 4, 5], 3],
        "output": 4
    }
]
```

Difference from the original code
---
* The range of embedding of integers
    * original: from -256 to 255, integers
    * this repository: real numbers
