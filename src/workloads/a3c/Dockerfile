# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND noninteractive

# Set the working directory to /app
WORKDIR /home

# Install any needed packages specified in requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt  # If you have any additional dependencies, list them in requirements.txt
#RUN cat requirements.txt | xargs pip install
RUN apt-get -y update && apt-get -y install python3-opencv 
RUN pip install opencv-python
RUN pip install setuptools==65.5.0
RUN pip install wheel==0.38.4
RUN pip install gym[atari,accept-rom-license]==0.21.0
RUN pip install setproctitle
RUN pip install grpcio-tools
RUN pip install dill


# CMD python main.py --env PongNoFrameskip-v4 --load --workers 4 --gpu-ids 0
# use CPU_COUNT env variable to set the number of workers
CMD cd /home && python /home/main.py --env PongNoFrameskip-v4 --load --workers $CPU_COUNT --episodes 3000000
# 1 for each worker and 1 for the test
