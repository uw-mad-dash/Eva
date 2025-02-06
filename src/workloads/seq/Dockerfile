FROM ubuntu:22.04

WORKDIR /home

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install grpcio-tools

CMD chmod +x ./diamond && bash run.sh