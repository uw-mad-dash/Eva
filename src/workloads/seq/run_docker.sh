#!/bin/bash

CPUS=8-15
CONTAINER_NAME=seq

sudo rm -rf ptmp/ tmp/
docker build -t $CONTAINER_NAME .
docker rm $CONTAINER_NAME
docker run --name $CONTAINER_NAME --cpuset-cpus $CPUS --shm-size 1m -v .:/home --mount type=bind,source=/home/ubuntu/mount/datasets,target=/datasets $CONTAINER_NAME
