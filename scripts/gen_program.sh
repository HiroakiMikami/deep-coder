#! /bin/sh

model=$1
example=$2
max_length=$3
method=$4

python3 $(dirname $0)/../python/predictor.py ${model} ${example} ${method} | sed -e 's/[ ]*$//g' | $(dirname $0)/../build/src/gen_program ${max_length} $method
exit $?