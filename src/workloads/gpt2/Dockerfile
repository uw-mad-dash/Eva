# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest

# Set the working directory to /app
WORKDIR /home

# COPY requirements.txt /home

# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt  # If you have any additional dependencies, list them in requirements.txt
RUN pip install regex
RUN pip install tqdm
RUN pip install torch
RUN pip install numpy
RUN pip install matplotlib
RUN pip install grpcio-tools

# print python version
# CMD pip list
CMD cd src && bash run.sh