# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest

# Set the working directory to /app
WORKDIR /home

# COPY requirements.txt /home

# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt  # If you have any additional dependencies, list them in requirements.txt
RUN pip install torchvision
RUN pip install torch
RUN pip install grpcio-tools

CMD cd /home && python image.py --data /datasets/imagenet-small/ --arch=resnet18 --batch-size=128 --epochs=6 --world-size=2 --rank=${WORKER_ID} --dist-url=tcp://${WORKER0_IP}:23422 --dist-backend='nccl' --checkpoint-dir /home --checkpoint-freq 32 --print-freq 32 --workers 32 --resume /home/checkpoint.pth.tar
# CMD python /home/image.py --dummy --arch=resnet18 --batch-size=4 --epochs=1 --world-size=2 --rank=1 --dist-url=tcp://${NODE0_IP}:23422 --dist-backend='nccl' --checkpoint-dir /home --checkpoint-freq 10 --resume /home/checkpoint.pth.tar
