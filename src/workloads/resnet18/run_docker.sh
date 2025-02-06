#!/bin/bash

GPUS=0
CPUS=0-3
CONTAINER_NAME=resnet18

rm -rf checkpoint.pth.tar
rm eva_iterator.log
docker build -t $CONTAINER_NAME .
docker rm $CONTAINER_NAME
docker run --name $CONTAINER_NAME --gpus all --runtime=nvidia --env NVIDIA_VISIBLE_DEVICES=$GPUS --env CUDA_VISIBLE_DEVICES=$GPUS  --cpuset-cpus $CPUS --shm-size 100g --mount type=bind,source=.,target=/home --mount type=bind,source=/home/ubuntu/mount/datasets,target=/datasets $CONTAINER_NAME
# docker run --gpus all --cpuset-cpus 0-3 -m 1g --shm-size 1g resnet50 python image.py -a resnet18 -b 32 --dummy
