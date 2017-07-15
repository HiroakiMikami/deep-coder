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
# Search program using a neural network model
$ ./scripts/gen_program.sh model/model.dat examples/dot.json 2
head    last    take    drop    access  minimum maximum reverse sort    sum     map     filter  count   zip_with        scanl1  >0      <0      %2 == 0 %2 == 1 +1      -1   *(-1)    *2      *3      *4      /2      /3      /4      **2     +       -       *       MIN     MAX # The probability of each functions
0.342   0.3     0.00922 0.37    0.492   0.212   0.514   0.192   0.0332  0.109   8.05e-06        0.000323        9.78e-05        0.00831 0.000241        0.00212 0.00347 5.23e-05      2.68e-06        0.00572 0.000305        0.00144 0.00061 0.000862        7.36e-05        0.00197 1.26e-05        0.0703  0.0371  0.84    0.0163  0.145   6.85e-07     6.85e-07 6.85e-07
--- # Program
a <- read_list
b <- read_list
c <- zip_with * b a
d <- sum c
---
$ ./scripts/gen_program.sh model/model.dat examples/dot.json 2 none # Search program without a neural network model
---
a <- read_list
b <- read_list
c <- zip_with * b a
d <- sum c
---
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
# Search program using a neural network model
$ ./scripts/gen_program.sh model.dat examples/reverse.json 1 2> /dev/null
---
a <- read_list
b <- reverse a
---
```

3. Experiment with Larger Dataset
---

### a. Dataset generation
Generate a dataset (N=about 200000):
```bash
$ ./build/src/gen_dataset 3 200000 10 2>&1 > dataset.json | grep Progress
```

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
Use CPU:
```bash
$ python3 ./python/learner.py dataset.json model.dat
```

Use GPU (`0` is a device id):
```bash
$ python3 ./python/learner.py dataset.json model.dat 0
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

TODO
---
* Over-fitting if program length >= 2
* Fasten dataset generation

Difference from the original code
---
* The range of embedding of integers
    * original: from -256 to 255, integers
    * this repository: real numbers
