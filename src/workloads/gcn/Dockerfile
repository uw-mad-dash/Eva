# Use an official PyTorch runtime as a parent image
FROM python:3.11.9

# Set the working directory to /app
WORKDIR /home

RUN pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/repo.html
RUN pip install torch==2.1.0
RUN pip install ogb
RUN pip install scikit-optimize>=0.9
RUN pip install --force-reinstall -v "scikit-learn==1.4.0"
RUN pip install grpcio-tools

CMD python main.py --dataset ogbn-products --sampler shadow --model gcn --epoch 100 --cpu_count $CPU_COUNT
