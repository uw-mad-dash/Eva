# Use an official PyTorch runtime as a parent image
FROM opencfd/openfoam-default

# Set the working directory to /app
WORKDIR /home

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install grpcio-tools

# CMD bash -c "cd /home && ./Allrun"
# CMD /openfoam/run -c "cd /home && ./Allrun"
#CMD ["/openfoam/run", "-c", "cd /home && chmod +x ./Allrun && chmod +x ./Allclean && ./Allrun"]
# CMD ["/openfoam/run", "-c", "cd /home && chmod +x Clean && ./Clean"]
# CMD ["/openfoam/run", "-c", "cd /home && ./Mesh 60 24 24"]
# CMD ["/openfoam/run", "-c", "cd /home && ./Setup 8"]
CMD ["/openfoam/run", "-c", "cd /home && chmod +x ./Solve && ./Solve"]
