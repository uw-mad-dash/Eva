# Use an official PyTorch runtime as a parent image
FROM nvcr.io/nvidia/dgl:23.07-py3

# Set the working directory to /app
WORKDIR /home

#COPY requirements.txt /home

# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt 
RUN pip install torch
RUN pip install ogb
RUN pip install dgl
RUN pip install tensorboardX
RUN pip install tqdm
RUN pip install grpcio-tools
# print python version
# CMD pip list
# CMD python cyclegan.py --n_cpus 4 --batch_size 4 --dataset_path /datasets/monet2photo --checkpoint_interval 150 --n_steps 2000 --decay_epoch 0
CMD python train.py --num_epoch 5
