#!/bin/bash

CPUS=17-26
CPU_COUNT=10

rm -rf trained_models/*
docker build --tag a3c .
docker rm a3c
rm -rf eva_iterator.log
# env variable CPU_COUNT
docker run --name a3c --cpuset-cpus $CPUS --shm-size 100g -v .:/home --env CPU_COUNT=$CPU_COUNT a3c
# docker run --gpus all --cpuset-cpus 0-31 a3c python main.py --env PongNoFrameskip-v4 --workers 32
