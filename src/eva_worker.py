import argparse
import json
import time
import os

from worker import Worker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="eva_config.json")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    worker = Worker(
        id=config["id"],
        ip_addr=config["ip_addr"],
        port=config["port"],
        master_ip_addr=config["master_ip_addr"],
        master_port=config["master_port"],
        swarm_ip_addr=config["swarm_ip_addr"],
        swarm_port=config["swarm_port"],
        swarm_token=config["swarm_token"],
        docker_network_name=config["docker_network"],
        iterator_port=config["iterator_port"],
        working_dir=os.path.expanduser(config["working_dir"]),
        datasets_dir=os.path.expanduser(config["datasets_dir"]),
        storage_manager_config=config["storage_manager"],
        mode=config["mode"],
        start_timestamp=float(config["start_timestamp"]),
    )


    try:
        print(f"Worker {config['id']} is up...")
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()