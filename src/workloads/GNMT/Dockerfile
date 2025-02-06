# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.06-py3
FROM ${FROM_IMAGE_NAME}

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN pip install --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git#egg=apex

WORKDIR /home

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pytablewriter==0.64.0
RUN pip install --no-cache-dir sacrebleu==1.2.10
RUN pip install --no-cache-dir sacremoses==0.0.19
RUN pip install --no-cache-dir pynvml==8.0.4
RUN pip install --no-cache-dir git+https://github.com/rsennrich/subword-nmt.git@48ba99e657591c329e0003f0c6e32e493fa959ef
RUN pip install --no-cache-dir git+https://github.com/NVIDIA/dllogger.git#egg=dllogger

CMD python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset-workers 4 --dataset-dir /datasets/wmt16_de_en --seed 2 --train-global-batch-size 1024 --resume checkpoint0.pth --save-freq 200 --keep-checkpoints 1 --epochs 2