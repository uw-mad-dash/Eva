#!/bin/bash
# aws s3 rm s3://eva-artifacts-eval/ --recursive
rm -rf ~/eva_report.json
sudo rm -rf ~/eva_worker*
pkill -f eva_worker
docker kill $(docker ps -q)
docker container prune -f
docker network prune -f
docker swarm leave --force
# sleep 3
sleep 3

python eva_master.py \
    --config-path $HOME"/eva/src/eva_config.json" 2>&1 | tee eva_master.log
