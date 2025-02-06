import subprocess
import os
import json
from itertools import product
import random

from parse_report.utils import calculate_total_cost

contention_factor = 0.95

# Function to generate a random port number
def generate_port(used_ports):
    port = None
    while port is None or port in used_ports:
        port = random.randint(20000,25000)
    return port

def generate_config(scheduler, trace_file, report_file, ports):
    config = {
        "verbose": False,
        "simulator_ip_addr": "localhost",
        "simulator_port": ports["simulator_port"],
        "event_receiver_ip_addr": "localhost",
        "event_receiver_port": ports["event_receiver_port"],
        "cloud_provisioner_config": "simulation/config/ec2_config_virt.json",
        "workload_trace_config": trace_file,
        "eva_ip_addr": "localhost",
        "eva_port": ports["eva_port"],
        "master_ip_addr": "localhost",
        "master_port": ports["master_port"],
        "worker_port": ports["worker_port"],
        "worker_working_dir": None,
        "mount_dir": None,
        "datasets_dir": None,
        "swarm_ip_addr": None,
        "swarm_port": None,
        "docker_subnet": None,
        "docker_iprange": None,
        "scheduling_interval": 300,
        "report_interval": 5000,
        "report_file": report_file,
        "mode": "simulation",
        "cloud_provisioner": {
            "class_name": "EC2Provisioner",
            "args": {
                "config_path": "simulation/config/ec2_config_virt.json"
            }
        },
        "storage_manager": None,
        "scheduler": {
            "class_name": scheduler,
            "args": {}
        },
        "ending_job_id": 9000,
        "iterator_port": None,
        "contention_factor": contention_factor,
        # "contention_map_file": None
        "contention_map_file": "simulation/contention_map/contention_map.pkl" 
    }
    if scheduler == "OwlScheduler":
        config["scheduler"]["args"]["contention_map_file"] = "simulation/contention_map/contention_map.pkl"
        # config["scheduler"]["args"]["contention_map_file"] = None
        # config["scheduler"]["args"]["contention_map_value"] = contention_factor
    return config

def run_experiment(working_dir, scheduler, trace_file, ports):
    # create a working directory
    os.makedirs(working_dir, exist_ok=True)
    log_file = os.path.join(working_dir, "log.txt")
    report_file = os.path.join(working_dir, "report.json")
    config_file = os.path.join(working_dir, "config.json")

    # first generate a config file
    config = generate_config(scheduler, trace_file, report_file, ports)
    with open(config_file, "w") as f:
        json.dump(config, f)

    command = [
        "python", "eva_simulate.py",
        "--config-path", config_file,
        "--logging"
    ]

    with open(log_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=f)

def main():
    schedulers = [
        "NaiveScheduler",
        "EVAGangScheduler",
        "StratusScheduler",
        "OwlScheduler",
        "SynergyScheduler",
    ]
    trace_files = [
        "pai_trace/traces/pai_full.json"
        ] 

    used_ports = set()

    experiments = list(product(schedulers, trace_files))

    for idx, (scheduler, trace_file) in enumerate(experiments):
        # Generate unique port numbers for each experiment
        ports = {}
        for port_name in ["simulator_port", "event_receiver_port", "eva_port", "master_port", "worker_port"]:
            ports[port_name] = generate_port(used_ports)
            used_ports.update([ports[port_name]])
        
        # trace_name is the last part of the trace file path
        trace_name = os.path.splitext(os.path.basename(trace_file))[0]
        # working dir prefix is the first part of the trace file path
        # working_dir_prefix = os.path.dirname(trace_file)
        working_dir_prefix = "simulation_experiments"
        working_dir = f"{working_dir_prefix}/{scheduler}_{trace_name}"
        # working_dir = f"{working_dir_prefix}/{scheduler}_{trace_name}"
        # working_dir = f"experiments_tmp/{scheduler}_{trace_name}"
        print(f"Running experiment {idx+1}/{len(experiments)}: {scheduler} {trace_file}")
        run_experiment(working_dir, scheduler, trace_file, ports)
        log_file = f"{working_dir}/log.txt"
        
        with open(log_file, "r") as f:
            lines = f.readlines()
            tail_lines = lines[-2:-1]
            for line in tail_lines:
                print(line.strip())

if __name__ == "__main__":
    main()
