#! /bin/bash
example_dir=$2
output_dir=$3

parallel "$(dirname $0)/do_test $1 $2 $3 $4 $5 {}" ::: $(seq 0 $(( $(ls ${example_dir}/*-example | wc -l) - 1)))
cat ${output_dir}/result-* > ${output_dir}/result
