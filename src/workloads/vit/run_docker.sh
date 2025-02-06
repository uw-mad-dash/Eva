#!/bin/bash

GPUS=0,1
CPUS=0-7
CONTAINER_NAME=vit

# rm -rf checkpoint.pth.tar
docker build -t $CONTAINER_NAME . 
docker rm $CONTAINER_NAME
docker run --name $CONTAINER_NAME --gpus all --runtime=nvidia --env NVIDIA_VISIBLE_DEVICES=$GPUS --env CUDA_VISIBLE_DEVICES=$GPUS  --cpuset-cpus $CPUS --shm-size 8g --mount type=bind,source=.,target=/home --mount type=bind,source=/home/ubuntu/mount/datasets,target=/datasets $CONTAINER_NAME
