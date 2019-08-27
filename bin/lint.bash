#! /bin/bash

find ./ ./src ./test -maxdepth 1 -name "*.py" | xargs -n 1 autopep8 --in-place
