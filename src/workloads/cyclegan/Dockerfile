# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest

# Set the working directory to /app
WORKDIR /home

# COPY requirements.txt /home

# Install any needed packages specified in requirements.txt
RUN pip install torchvision
RUN pip install grpcio-tools
# RUN pip install --no-cache-dir -r requirements.txt  # If you have any additional dependencies, list them in requirements.txt
# print python version
# CMD pip list
CMD python cyclegan.py --num_workers 8 --batch_size 4 --dataset_path /datasets/monet2photo --checkpoint_interval 20 --n_steps 8000000 --decay_epoch 0
