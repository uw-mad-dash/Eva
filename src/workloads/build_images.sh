#!/bin/bash
docker image build -t sage sage/
docker image build -t resnet18 resnet18/
docker image build -t a3c a3c/
docker image build -t gpt2 gpt2/
docker image build -t gcn gcn/
docker image build -t seq seq/
docker image build -t cyclegan cyclegan/
docker image build -t openfoam openfoam/
docker image build -t vit vit/
