#!/bin/bash

# docker build --tag node1 .
docker run --cpuset-cpus 0-3 --shm-size 100g -v .:/home --net job_0 --ip 10.0.1.3 node1
# docker run --gpus all --cpuset-cpus 0-3 -m 1g --shm-size 1g resnet50 python image.py -a resnet18 -b 32 --dummy
