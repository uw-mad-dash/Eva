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

CMD cd /home && python image.py -a resnet18 -b 128 --data /datasets/imagenet/ --arch=resnet18 --checkpoint-dir /home --checkpoint-freq 32 --print-freq 32 --workers 32 --epochs 1 --resume /home/checkpoint.pth.tar
