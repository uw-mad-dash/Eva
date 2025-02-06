#!/bin/bash

CPUS=0-3
GPUS=0
CONTAINER_NAME=cyclegan

rm -rf model.chkpt
docker build -t $CONTAINER_NAME .
docker rm $CONTAINER_NAME
docker run --privileged --name $CONTAINER_NAME --gpus all --runtime=nvidia --env NVIDIA_VISIBLE_DEVICES=$GPUS --env CUDA_VISIBLE_DEVICES=$GPUS --cpuset-cpus $CPUS -m 10g --shm-size 1g --mount type=bind,source=.,target=/home --mount type=bind,source=/home/ubuntu/mount/datasets,target=/datasets $CONTAINER_NAME
