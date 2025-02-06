import argparse
from master import Master
from submission_manager import SubmissionManager
import json
import time
import socket
import os

def main():
    default_ip_addr = socket.gethostbyname(socket.gethostname())

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="eva_config.json")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    mode = config["mode"]
    config["cloud_provisioner"]["args"]["mode"] = mode

    master = Master(
        ip_addr=config["master_ip_addr"],
        port=config["master_port"],
        worker_port=config["worker_port"],
        worker_working_dir=config["worker_working_dir"],
        mount_dir=config["mount_dir"],
        datasets_dir=config["datasets_dir"],
        swarm_ip_addr=config["swarm_ip_addr"],
        swarm_port=config["swarm_port"],
        docker_subnet=config["docker_subnet"],
        docker_iprange=config["docker_iprange"],
        iterator_port=config["iterator_port"],
        scheduling_interval=config["scheduling_interval"],
        report_interval=config["report_interval"],
        report_file=os.path.expanduser(config["report_file"]),
        cloud_provisioner_config=config["cloud_provisioner"],
        storage_manager_config=config["storage_manager"],
        scheduler_config=config["scheduler"],
        mode=config["mode"],
        verbose=config["verbose"],
    )

    submission_manager = SubmissionManager(
        server_ip_addr=config["eva_ip_addr"],
        server_port=config["eva_port"],
        callbacks=master.get_submission_manager_callbacks())
    
    try:
        print("EVA is up...")
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        print("Exiting...")

    
if __name__ == "__main__":
    main()