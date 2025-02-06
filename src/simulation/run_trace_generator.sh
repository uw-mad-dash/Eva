#!/bin/bash

python3 trace_generator.py \
    --num-jobs 20 \
    --ending-job-id 31 \
    --avail-time-lambda 10 \
    --workload-catalog-path "workload_catalog.json" \
    --output-path "small_scale/small_fidelity.json" \
    --seed 3
