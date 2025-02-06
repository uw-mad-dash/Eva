#!/bin/bash

python eva_simulate.py \
    --config-path $HOME"/eva/src/simulation/config/simulation_config.json" 2>&1 \
    --logging \
    | tee eva_simulation.log
