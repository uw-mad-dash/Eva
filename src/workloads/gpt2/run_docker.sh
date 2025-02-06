#!/bin/bash

CONTAINER_NAME=gpt2
GPUS=0,1,2,3
CPUS=0-7

# rm -rf src/ckpt-gpt2.pth
docker build -t $CONTAINER_NAME .
docker rm $CONTAINER_NAME; docker run                 --privileged                 --name $CONTAINER_NAME               --cpuset-cpus $CPUS                 --memory 10g                 --shm-size 1g                 --volume .:/home:rw                 --mount type=bind,source=/home/ubuntu/mount/datasets,target=/datasets,readonly --gpus all --runtime=nvidia --env NVIDIA_VISIBLE_DEVICES=$GPUS --env CUDA_VISIBLE_DEVICES=$GPUS --env NODE0_IP=10.0.0.2 $CONTAINER_NAME
