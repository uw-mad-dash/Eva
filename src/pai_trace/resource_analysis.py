import numpy as np
import json
import random
import argparse
import pathlib
import csv
import math
import matplotlib.pyplot as plt

class WorkloadTask:
    def __init__(self, name, demand, shm_size, fetch_delay, build_image_delay, kill_delay, upload_delay, image_id):
        self.name = name
        self.demand = demand
        self.shm_size = shm_size
        self.fetch_delay = fetch_delay
        self.build_image_delay = build_image_delay
        self.kill_delay = kill_delay
        self.upload_delay = upload_delay
        self.image_id = image_id # basically workload_task_id

class Workload:
    def __init__(self, name, workload_tasks, init_delay, full_throughput, support_throughput_aware, duration):
        self.name = name
        self.workload_tasks = workload_tasks # a list of WorkloadTask
        self.init_delay = init_delay
        self.full_throughput = full_throughput
        self.support_throughput_aware = support_throughput_aware
        self.duration = duration
    
    def __str__(self):
        res = f"Workload \n"
        # loop through attributes
        for attr, value in self.__dict__.items():
            res += f"{attr}: {value}\n"
        return res

class PAIJob:
    def __init__(self, name, gpu, cpu, memory, creation_time, deletion_time):
        self.name = name
        self.gpu = gpu
        self.cpu = cpu
        self.memory = memory
        self.creation_time = creation_time
        self.deletion_time = deletion_time
        self.duration = deletion_time - creation_time

def read_pai_jobs(path, ec2_config_path):
    instance_type_capacity = {}
    with open(ec2_config_path, 'r') as file:
        ec2_config = json.load(file)
        for instance_type, instance_desc in ec2_config["instance_types"].items():
            instance_type_capacity[instance_type] = {
                "gpu": instance_desc["capacity"][0],
                "cpu": instance_desc["capacity"][1],
                "memory": instance_desc["capacity"][2]
            }

    pai_jobs = []
    with open(path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            name = row['name']

            num_gpu = float(row['gpu_milli'])*float(row['num_gpu'])
            num_cpu = float(row['cpu_milli'])
            memory_gb = math.ceil(int(row['memory_mib']) / 1024)
            creation_time = int(row['creation_time'])
            deletion_time = int(row['deletion_time'])

            pai_job = PAIJob(name, num_gpu, num_cpu, memory_gb, creation_time, deletion_time)
            pai_jobs.append(pai_job)
        
    return pai_jobs

def main():
    file_dir = pathlib.Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-jobs", type=int, default=500)
    parser.add_argument("--ending-job-id", type=int, default=29)
    parser.add_argument("--avail-time-lambda", type=int, default=10800)
    parser.add_argument("--workload-catalog-path", type=str, default=f"{file_dir}/../workloads/workload_catalog.json")
    parser.add_argument("--output-path", type=str, default=f'{file_dir}/traces/5hr_interarrival.json')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ec2-config-path", type=str, default=f"{file_dir}/../ec2_config.json")
    parser.add_argument("--pai-jobs-path", type=str, default=f"{file_dir}/openb_pod_list_default.csv")
    parser.add_argument("--use-pai-arrival-time", type=bool, default=False)
    parser.add_argument("--use-pai-duration", type=bool, default=False)

    args = parser.parse_args()
    random.seed(args.seed)
    pai_jobs = read_pai_jobs(args.pai_jobs_path, args.ec2_config_path)

    # # print number of distinct job demands
    # print(len(set([(job.gpu, job.cpu, job.memory) for job in pai_jobs])))
    # for job in list(set([(job.gpu, job.cpu, job.memory) for job in pai_jobs])):
    #     print(job)
    
    # Filter single GPU jobs
    single_gpu_jobs = [job for job in pai_jobs if job.gpu > 0]
    print(len(single_gpu_jobs))
    
    # Extract CPU and memory demands
    cpu_demands = [job.cpu for job in single_gpu_jobs]
    memory_demands = [job.memory for job in single_gpu_jobs]

    gpu_demands = [job.gpu for job in single_gpu_jobs]
    # cpu_demands = [job.cpu for job in pai_jobs]
    # memory_demands = [job.memory for job in pai_jobs]

    # print coeeficient of correlation of three demands
# Create a matrix from the demands
    demands_matrix = np.array([gpu_demands, cpu_demands, memory_demands])

    # Compute the correlation matrix
    correlation_matrix = np.corrcoef(demands_matrix)

    # Print the correlation matrix
    print(correlation_matrix)
    return

    # correlation between cpu and memory demands
    
    # Plot the CPU and memory demands for single GPU jobs as a heatmap
    plt.figure(figsize=(10, 7))
    plt.hexbin(cpu_demands, memory_demands, gridsize=50, cmap='Grays')
    plt.colorbar(label='Number of Jobs')
    plt.xlabel('CPU Demand (vCPUs)')
    plt.ylabel('Memory Demand (GB)')
    plt.title('CPU and Memory Demands for Single GPU Jobs')
    plt.grid(True)
    plt.savefig(f'{file_dir}/demand_vectors.png')

if __name__ == "__main__":
    main()
