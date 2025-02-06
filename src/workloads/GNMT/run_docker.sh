#!/bin/bash

# docker run --gpus all -it --cpuset-cpus 0-15 --shm-size 100g -v /home/ubuntu/mount/workspace/GNMT:/home -v /home/ubuntu/mount/datasets:/datasets gnmt
docker run --gpus all -it --cpuset-cpus 0-15 --shm-size 100g -v /home/ubuntu/GNMT:/home -v /home/ubuntu/mount/datasets:/datasets gnmt

python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset-dir /datasets/wmt16_de_en --seed 2 --train-global-batch-size 1024 --resume gnmt/checkpoint0.pth --save-freq 200 --keep-checkpoints 1 