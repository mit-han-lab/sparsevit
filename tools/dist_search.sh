#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
MAX=$3
MIN=$4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
    $(dirname "$0")/search.py $CONFIG $CHECKPOINT --max $MAX --min $MIN --launcher pytorch ${@:4}
