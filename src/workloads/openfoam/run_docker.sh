#!/bin/bash

CPUS=8-15
CONTAINER_NAME=openfoam

sudo rm -rf log.simpleFoam 
sudo env "PATH=$PATH" python reset.py
docker build -t $CONTAINER_NAME .
docker rm $CONTAINER_NAME
docker run --name $CONTAINER_NAME --cpuset-cpus $CPUS --shm-size 1g -v .:/home --mount type=bind,source=/home/ubuntu/mount/datasets,target=/datasets $CONTAINER_NAME

