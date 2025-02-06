#!/bin/bash

# docker build --tag node0 .
docker run --cpuset-cpus 0-3 --shm-size 100g -v .:/home --net eva_network --ip 10.0.0.1 -it task_0
# docker run --gpus all --cpuset-cpus 0-3 -m 1g --shm-size 1g resnet50 python image.py -a resnet18 -b 32 --dummy
