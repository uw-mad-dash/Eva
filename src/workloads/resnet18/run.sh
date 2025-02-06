#!/bin/bash

python image.py -a resnet18 -b 128 --dummy --checkpoint-dir /checkpoint --checkpoint-freq 100 --resume /checkpoint/checkpoint.pth.tar
