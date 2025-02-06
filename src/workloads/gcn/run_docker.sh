#!/bin/bash

# docker build --tag resnet50 .
# docker run -it --cpuset-cpus 0-7 --shm-size 100g -v .:/home --mount type=bind,source=/home/ubuntu/mount/datasets,target=/datasets argo /bin/bash
CPUS=17-28
CPU_COUNT=12

rm -rf model.pt
docker build --tag gcn .
docker rm gcn
docker run --name gcn --cpuset-cpus $CPUS --shm-size 8g -v .:/home --mount type=bind,source=/home/ubuntu/mount/datasets,target=/datasets --env CPU_COUNT=$CPU_COUNT gcn 
# docker rm task_0; docker container create                 --privileged                 --name task_0                 --network eva_bridge                 --ip 192.0.0.1                 --cpuset-cpus $CPUS                 --memory 40g                 --shm-size 8g                 --volume .:/home:rw                 --mount type=bind,source=/home/ubuntu/mount/datasets,target=/datasets,readonly --env NODE0_IP=10.0.0.1 --env CPU_COUNT=6 --env EVA_JOB_ID=0 --env EVA_TASK_ID=0 --env EVA_WORKER_IP_ADDR=localhost --env EVA_WORKER_PORT=60000 --env EVA_ITERATOR_IP_ADDR=192.0.0.1 --env EVA_ITERATOR_PORT=50425 task_0; docker network connect eva_network --ip 10.0.0.1 task_0; docker start task_0

