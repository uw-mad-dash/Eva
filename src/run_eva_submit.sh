#!/bin/bash
EVA_IP_ADDR="172.31.17.248"
EVA_PORT=50422
LOCAL_WORKING_DIR=$HOME"/eva/src/workloads/two_node_Resnet18"
# LOCAL_WORKING_DIR=$HOME"/subme"
# GLOBAL_WORKING_DIR="workspace/job_gcnnn2"
# LOCAL_WORKING_DIR=$HOME"/eva/src/examplar_jobs/gpt2"
# GLOBAL_WORKING_DIR="workspace/job_gasffft2"
# LOCAL_WORKING_DIR=$HOME"/seq"
GLOBAL_WORKING_DIR="workspace/2node"

rm -rf ~/mount/$GLOBAL_WORKING_DIR

python eva_submit.py \
    --eva-ip-addr $EVA_IP_ADDR \
    --eva-port $EVA_PORT \
    --local-working-dir $LOCAL_WORKING_DIR \
    --global-working-dir $GLOBAL_WORKING_DIR

# LOCAL_WORKING_DIR=$HOME"/eva/src/examplar_jobs/gpt2"
# GLOBAL_WORKING_DIR="workspace/gtat2"
# python eva_submit.py \
#     --eva-ip-addr $EVA_IP_ADDR \
#     --eva-port $EVA_PORT \
#     --local-working-dir $LOCAL_WORKING_DIR \
#     --global-working-dir $GLOBAL_WORKING_DIR


