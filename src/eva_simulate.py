import argparse
from master import Master
from submission_manager import SubmissionManager
from simulator import Simulator
import json
import logging.config
import time
import socket
import os

def main():
    default_ip_addr = socket.gethostbyname(socket.gethostname())

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="simulation_config.json")
    parser.add_argument("--logging", action="store_true")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    mode = config["mode"]
    config["cloud_provisioner"]["args"]["mode"] = mode

    simulator = Simulator(
        ip_addr=config["simulator_ip_addr"],
        port=config["simulator_port"],
        eva_ip_addr=config["eva_ip_addr"],
        eva_port=config["eva_port"],
        master_ip_addr=config["master_ip_addr"],
        master_port=config["master_port"],
        event_receiver_ip_addr=config["event_receiver_ip_addr"],
        event_receiver_port=config["event_receiver_port"],
        scheduling_interval=config["scheduling_interval"],
        cloud_provisioner_config=config["cloud_provisioner_config"],
        workload_trace_config=config["workload_trace_config"],
        ending_job_id=config["ending_job_id"],
        contention_factor=config["contention_factor"],
        contention_map_file=config["contention_map_file"],
        # event_delay_config=config["event_delay_config"],
    )

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
        simulator_ip_addr=config["simulator_ip_addr"],
        simulator_port=config["simulator_port"],
        simulation_event_receiver_ip_addr=config["event_receiver_ip_addr"],
        simulation_event_receiver_port=config["event_receiver_port"],
        verbose=config["verbose"],
    )

    submission_manager = SubmissionManager(
        server_ip_addr=config["eva_ip_addr"],
        server_port=config["eva_port"],
        callbacks=master.get_submission_manager_callbacks())
    
    # logging
    if not args.logging:
        logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True
    })
    
    simulator.run()

    master.shut_down()

    
if __name__ == "__main__":
    main()