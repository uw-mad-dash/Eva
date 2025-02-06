#!/bin/bash

# python image.py -a resnet18 -b 128 --dummy --checkpoint-dir /checkpoint --checkpoint-freq 100 --resume /checkpoint/checkpoint.pth.tar
# python image.py --dummy --arch=resnet18 --batch-size=128 --epochs=1 --world-size=2 --rank=0 --dist-url='tcp://172.31.34.175:23456' --dist-backend='gloo' --multiprocessing-distributed
# python image.py --dummy --arch=resnet18 --batch-size=128 --epochs=1 --world-size=2 --rank=1 --dist-url='tcp://172.31.34.175:23456' --dist-backend='gloo' --multiprocessing-distributed
PORT=23422
python image.py --dummy --arch=resnet18 --batch-size=4 --epochs=1 --world-size=2 --rank=0 --dist-url='tcp://127.0.0.1:'$PORT --dist-backend='gloo' --multiprocessing-distributed
python image.py --dummy --arch=resnet18 --batch-size=4 --epochs=1 --world-size=2 --rank=1 --dist-url='tcp://127.0.0.1:'$PORT --dist-backend='gloo' --multiprocessing-distributed
