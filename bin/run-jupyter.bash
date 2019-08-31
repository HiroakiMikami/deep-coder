#! /bin/bash

set -u

jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=${1:-8888} \
  --NotebookApp.port_retries=0 \
  --no-browser
