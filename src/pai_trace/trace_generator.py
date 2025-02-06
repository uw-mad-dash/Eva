import numpy as np
import json
import random
import argparse
import pathlib
import csv
import math

class WorkloadTask:
    def __init__(self, name, demand, shm_size, full_throughput, fetch_delay, build_image_delay, kill_delay, upload_delay, image_id):
        self.name = name
        self.demand = demand
        self.shm_size = shm_size
        self.full_throughput = full_throughput
        self.fetch_delay = fetch_delay
        self.build_image_delay = build_image_delay
        self.kill_delay = kill_delay
        self.upload_delay = upload_delay
        self.image_id = image_id # basically workload_task_id

class Workload:
    def __init__(self, name, workload_tasks, init_delay, support_throughput_aware, duration):
        self.name = name
        self.workload_tasks = workload_tasks # a list of WorkloadTask
        self.init_delay = init_delay
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

class TraceGenerator:
    def __init__(self, pai_jobs, ending_job_id, avail_time_lambda, workload_catalog_path, 
                 seed,
                 use_pai_arrival_time=False,
                 use_pai_duration=False,
                 random_migration_time=False,
                 workload_weights=None,
                 multi_task_chance=0.0):
        self.pai_jobs = pai_jobs
        self.ending_job_id = ending_job_id
        self.avail_time_lambda = avail_time_lambda
        self.workload_catalog = {} # workload_name -> Workload
        self.seed = seed
        self.use_pai_arrival_time = use_pai_arrival_time
        self.use_pai_duration = use_pai_duration
        self.multi_task_chance = multi_task_chance
        self.read_workload_catalog(workload_catalog_path, random_migration_time)
        
        self.workload_weights = workload_weights # workload_name -> weight

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        
    def read_workload_catalog(self, path, rand_migration_time=False):
        self.set_seed(self.seed)

        with open(path, 'r') as file:
            catalog = json.load(file)

        image_id = 0
        for _, workload_desc in catalog.items():
            workload_name = workload_desc['name']
            workload_init_delay = workload_desc['init_delay']
            workload_duration = workload_desc['duration'] if workload_name == "seq" else None
            workload_support_throughput_aware = workload_desc['support_throughput_aware']
            workload_tasks = []
            if len(workload_desc['tasks']) > 1:
                continue
            for task_desc in workload_desc['tasks']:
                task_name = task_desc['name']
                demand = task_desc['demand']
                shm_size = task_desc['shm_size']
                full_throughput = task_desc['full_throughput']
                if rand_migration_time:
                    fetch_delay = random.randint(1, 60)
                    build_image_delay = random.randint(120, 180)
                    kill_delay = 1
                    upload_delay = random.randint(1, 60)
                else:
                    # assuming single worker jobs
                    fetch_delay = workload_desc['fetch_delay']
                    build_image_delay = workload_desc['build_image_delay']
                    kill_delay = workload_desc['kill_delay']
                    upload_delay = workload_desc['upload_delay']
                workload_task = WorkloadTask(
                    task_name, 
                    demand, 
                    shm_size,
                    full_throughput,
                    fetch_delay, 
                    build_image_delay,
                    kill_delay,
                    upload_delay,
                    image_id)
                workload_tasks.append(workload_task)

                image_id += 1
            
            workload = Workload(
                name=workload_name, 
                workload_tasks=workload_tasks, 
                init_delay=workload_init_delay, 
                support_throughput_aware=workload_support_throughput_aware,
                duration=workload_duration)
            self.workload_catalog[workload_name] = workload


            print(workload)
    
    def choose_workload(self):
        if self.workload_weights is None:
            return random.choice(list(self.workload_catalog.keys()))
        else:
            return np.random.choice(list(self.workload_catalog.keys()), p=self.workload_weights)
        
    def generate(self):
        task_id = 0
        trace = {} # job_id -> {}
        num_jobs = len(self.pai_jobs)

        prev_job_arrival_time = 0
        self.set_seed(self.seed)
        self.set_seed(self.seed)
        job_arrival_times = np.cumsum(np.random.exponential(self.avail_time_lambda, num_jobs))
        self.set_seed(self.seed)
        durations = [int(10 ** random.uniform(1.5, 3) * 60) if random.random() <= 0.8 else int(10 ** random.uniform(3, 4) * 60) for _ in range(num_jobs)]
        # durations = [int(10 ** random.uniform(1.5, 2.5) * 60) for _ in range(self.num_jobs)]

        num_distinct_pai_jobs = len(set([(job.gpu, job.cpu, job.memory) for job in self.pai_jobs]))
        # match each distinct pai job with a workload
        # name must be unique, so if say two jobs are both mapped to ResNet, they will call ResNet[1] and ResNet[2]
        pai_job_resource_demand_to_workload_name = {}
        workload_counter = {}
        for i, job in enumerate(self.pai_jobs):
            if (job.gpu, job.cpu, job.memory) not in pai_job_resource_demand_to_workload_name:
                # randomly select a workload
                chosen_workload_name = self.choose_workload()
                if chosen_workload_name not in workload_counter:
                    workload_counter[chosen_workload_name] = -1
                workload_counter[chosen_workload_name] += 1
                workload_name = f"{chosen_workload_name}[{workload_counter[chosen_workload_name]}]"
                pai_job_resource_demand_to_workload_name[(job.gpu, job.cpu, job.memory)] = workload_name


        for job_id in range(num_jobs):
            # Generate job arrival time
            if self.use_pai_arrival_time:
                job_arrival_time = self.pai_jobs[job_id].creation_time
            else:
                job_arrival_time = int(job_arrival_times[job_id])

            # Generate job
            workload_name = pai_job_resource_demand_to_workload_name[(self.pai_jobs[job_id].gpu, self.pai_jobs[job_id].cpu, self.pai_jobs[job_id].memory)]
            # actual workload name is workload_name.split("[")[0]
            workload = self.workload_catalog[workload_name.split("[")[0]]
            
            # job duration
            if self.use_pai_duration:
                duration = self.pai_jobs[job_id].duration
            else:
                # duration = durations[job_id] if workload.duration is None else workload.duration
                duration = durations[job_id]


            init_delay = workload.init_delay
            full_throughput = workload.workload_tasks[0].full_throughput
            support_throughput_aware = workload.support_throughput_aware

            total_iters = int(duration * full_throughput)
            
            job_desc = {
                "name": workload_name,
                "arrival_time": job_arrival_time,
                "duration": duration,
                "init_delay": init_delay,
                "full_throughput": full_throughput,
                "total_iters": total_iters,
                "support_throughput_aware": support_throughput_aware,
                "tasks": {}
            }
            for workload_task in workload.workload_tasks:
                demand_dict = {}
                demand_dict = {"any": [self.pai_jobs[job_id].gpu, self.pai_jobs[job_id].cpu, self.pai_jobs[job_id].memory]}
                # if self.pai_jobs[job_id].gpu > 0:
                #     demand_dict = {"p3": [self.pai_jobs[job_id].gpu, self.pai_jobs[job_id].cpu, self.pai_jobs[job_id].memory]}
                # else:
                #     demand_dict = {
                #         "p3": [0, self.pai_jobs[job_id].cpu, self.pai_jobs[job_id].memory],
                #         "c7i": [0, max(1, int(self.pai_jobs[job_id].cpu / 2)), self.pai_jobs[job_id].memory],
                #         "r7i": [0, max(1, int(self.pai_jobs[job_id].cpu / 2)), self.pai_jobs[job_id].memory]
                #     }
                # if random.random() < self.multi_task_chance:
                #     # randomly choose from 2 and 4  
                #     num_tasks = random.choice([2, 4, 8])
                # else:
                #     num_tasks = 1
                # num_task has {1,2,4,8} with prob {4, 3, 2, 1}
                num_tasks = np.random.choice([1, 1], p=[0.55, 0.45])
                for i in range(num_tasks):
                    job_desc["tasks"][task_id] = {
                        "name": workload_task.name,
                        "demand": demand_dict,
                        "shm_size": workload_task.shm_size,
                        "image_id": workload_task.image_id,
                        "fetch_delay": workload_task.fetch_delay,
                        "build_image_delay": workload_task.build_image_delay,
                        "kill_delay": workload_task.kill_delay,
                        "upload_delay": workload_task.upload_delay
                    }
                    task_id += 1
                # consists only of one task
                break

            trace[job_id] = job_desc
        
        workload_name_to_count = {}
        for job_id, job_desc in trace.items():
            workload_name = job_desc["name"]
            if workload_name not in workload_name_to_count:
                workload_name_to_count[workload_name] = 0
            workload_name_to_count[workload_name] += 1
        
        # print workload name, count, demand
        for workload_name, count in workload_name_to_count.items():
            # find the resource demand
            resource_demand = None
            for job_id, job_desc in pai_job_resource_demand_to_workload_name.items():
                if job_desc == workload_name:
                    resource_demand = job_id
                    break
            print(f"Workload: {workload_name}, count: {count}, demand: {resource_demand}")
        
        # check that the arrival time of the last job is way after the start time
        # + duration of the ending_job_id

        # if self.ending_job_id is not None and \
        #     trace[self.num_jobs-1]["arrival_time"] < trace[self.ending_job_id]["arrival_time"] + trace[self.ending_job_id]["duration"] + 10000:
        #     raise ValueError("The arrival time of the last job is not way after the start time + duration of the ending_job_id")

        # temporary: use job 2000 -- 2000 + self.max_num_jobs
        # make task_id start from 0, arrival_time start from 1
        
        return trace
    
    def generate_to_file(self, output_path):
        trace = self.generate()
        with open(output_path, 'w') as file:
            json.dump(trace, file, indent=4)
        return trace

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

            num_gpu = math.ceil(float(row['num_gpu']))
            num_cpu = math.ceil(int(row['cpu_milli']) / 1000)
            memory_gb = math.ceil(int(row['memory_mib']) / 1024)
            creation_time = int(row['creation_time'])
            deletion_time = int(row['deletion_time'])
            status = row['pod_phase']
            if status == "Failed":
                continue
            # make sure its demand is within the capacity of at least one instance type
            # otherwise skip
            can_be_accommodated = False
            for instance_type, capacity in instance_type_capacity.items():
                if num_gpu > 0:
                    if num_gpu <= capacity["gpu"] and num_cpu <= capacity["cpu"] and memory_gb <= capacity["memory"]:
                        can_be_accommodated = True
                        break
                else:
                    if capacity["gpu"] > 0 and num_cpu <= capacity["cpu"] and memory_gb <= capacity["memory"]:
                        can_be_accommodated = True
                        break
                    if capacity["gpu"] == 0 and max(1, int(num_cpu / 2)) <= capacity["cpu"] and memory_gb <= capacity["memory"]:
                        can_be_accommodated = True
                        break
            if not can_be_accommodated:
                continue

            pai_job = PAIJob(name, num_gpu, num_cpu, memory_gb, creation_time, deletion_time)
            pai_jobs.append(pai_job)
        
    return pai_jobs


def main():
    file_dir = pathlib.Path(__file__).parent
    parser = argparse.ArgumentParser()
    job_arrival_rate = 3 # 3 == 1 job every 20 minutes
    default_arrival_time = int(60 * 60 / job_arrival_rate)
    parser.add_argument("--num-jobs", type=int, default=500)
    parser.add_argument("--ending-job-id", type=int, default=29)
    parser.add_argument("--avail-time-lambda", type=int, default=default_arrival_time)
    parser.add_argument("--workload-catalog-path", type=str, default=f"{file_dir}/../workloads/workload_catalog.json")
    # parser.add_argument("--workload-weights", type=float, nargs='*')
    # parser.add_argument("--output-path", type=str, default=f'{file_dir}/../resubmission_simulation_experiments/end_to_end/default.json')
    # parser.add_argument("--output-path", type=str, default=f'{file_dir}/../resubmission_simulation_experiments/vary_gpu_virt/default_{job_arrival_rate}.json')
    parser.add_argument("--output-path", type=str, default=f'{file_dir}/../resubmission_simulation_experiments/multi_task_virt_diff_task/default_1.json')
    # parser.add_argument("--output-path", type=str, default=f'{file_dir}/traces/480min_interarrival.json')
    parser.add_argument("--seed", type=int, default=2)
    # parser.add_argument("--ec2-config-path", type=str, default=f"{file_dir}/../ec2_config.json")
    parser.add_argument("--ec2-config-path", type=str, default=f"{file_dir}/../simulation/config/ec2_config_virt.json")
    parser.add_argument("--pai-jobs-path", type=str, default=f"{file_dir}/openb_pod_list_default.csv")
    parser.add_argument("--use-pai-arrival-time", type=bool, default=False)
    parser.add_argument("--use-pai-duration", type=bool, default=False)
    parser.add_argument("--multi-task-chance", type=float, default=0.8)

    args = parser.parse_args()
    random.seed(args.seed)
    pai_jobs = read_pai_jobs(args.pai_jobs_path, args.ec2_config_path)
    # randomly sample 500
    # pai_jobs = random.sample(pai_jobs, 500)
    random.shuffle(pai_jobs)
    pai_jobs = pai_jobs[:200]
    # normalize time for the first job to start at 1
    # start_time = pai_jobs[1000].creation_time - 1
    # for job in pai_jobs:
    #     job.creation_time -= start_time
    #     job.deletion_time -= start_time
    trace_generator = TraceGenerator(
        pai_jobs=pai_jobs,
        ending_job_id=args.ending_job_id,
        avail_time_lambda=args.avail_time_lambda,
        workload_catalog_path=args.workload_catalog_path,
        random_migration_time=False,
        seed=args.seed,
        use_pai_arrival_time=args.use_pai_arrival_time,
        use_pai_duration=args.use_pai_duration,
        multi_task_chance=args.multi_task_chance
        # workload_weights=args.workload_weights
    )
    trace = trace_generator.generate_to_file(args.output_path)

    print(f"job arrival rate: {job_arrival_rate} (every {default_arrival_time / 60} minutes)")
    print(f"Number of PAI jobs: {len(pai_jobs)}")
    print(f"Percentage of no GPU jobs: {len([job for job in pai_jobs if job.gpu == 0]) / len(pai_jobs)}")
    print(f"Percentage of 1 GPU jobs: {len([job for job in pai_jobs if job.gpu == 1]) / len(pai_jobs)}")
    print(f"Percentage of 2 GPU jobs: {len([job for job in pai_jobs if job.gpu == 2]) / len(pai_jobs)}")
    print(f"Percentage of 4 GPU jobs: {len([job for job in pai_jobs if job.gpu == 4]) / len(pai_jobs)}")
    print(f"Percentage of 8 GPU jobs: {len([job for job in pai_jobs if job.gpu == 8]) / len(pai_jobs)}")
    print(f"Percentage of 1-task jobs: {len([job_id for job_id in trace if len(trace[job_id]['tasks']) == 1]) / len(trace)}")
    print(f"Percentage of 2-task jobs: {len([job_id for job_id in trace if len(trace[job_id]['tasks']) == 2]) / len(trace)}")
    print(f"Percentage of 4-task jobs: {len([job_id for job_id in trace if len(trace[job_id]['tasks']) == 4]) / len(trace)}")
    print(f"Percentage of 8-task jobs: {len([job_id for job_id in trace if len(trace[job_id]['tasks']) == 8]) / len(trace)}")
    # print(f"percentage of 2-task jobs: {len([job for job in pai_jobs if len(job.tasks) == 2]) / len(pai_jobs)}")
    # print(f"percentage of 4-task jobs: {len([job for job in pai_jobs if len(job.tasks) == 4]) / len(pai_jobs)}")


if __name__ == "__main__":
    main()

                

    