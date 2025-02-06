#!/bin/bash

GPUS=0
CPUS=0-7
CONTAINER_NAME=sage

rm -rf latest.ckpt
docker build -t $CONTAINER_NAME .
docker rm $CONTAINER_NAME; docker run                 --privileged                 --name $CONTAINER_NAME                               --cpuset-cpus $CPUS               --memory 400g                 --shm-size 32g                 --volume .:/home:rw                 --mount type=bind,source=/home/ubuntu/mount/datasets,target=/datasets,readonly --gpus all --runtime=nvidia --env NVIDIA_VISIBLE_DEVICES=$GPUS --env CUDA_VISIBLE_DEVICES=$GPUS --env NODE0_IP=10.0.0.1 $CONTAINER_NAME
