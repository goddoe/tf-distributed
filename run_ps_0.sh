#!/usr/bin/env bash
python3 run.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=localhost:2223 \
     --job_name=ps --task_index=0 \
     --dataset=./data/titanic/preprocessed/titanic.csv \
     --output=./output \
     --param_path=./param.json
