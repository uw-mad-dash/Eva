import numpy as np
import json
import random
import argparse
import pathlib

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

class TraceGenerator:
    def __init__(self, num_jobs, ending_job_id, avail_time_lambda, workload_catalog_path, 
                 seed,
                 random_migration_time=False,
                 workload_weights=None,
                 workload_counts=None):
        self.num_jobs = num_jobs
        self.ending_job_id = ending_job_id
        self.avail_time_lambda = avail_time_lambda
        self.workload_catalog = {} # workload_name -> Workload
        self.seed = seed
        self.read_workload_catalog(workload_catalog_path, random_migration_time)
        
        self.workload_weights = workload_weights # workload_name -> weight
        self.workload_counts = workload_counts # workload_name -> count

        if self.workload_weights is not None:
            total_weight = sum(self.workload_weights.values())
            for key in self.workload_weights:
                self.workload_weights[key] /= total_weight
        
        if self.workload_counts is not None:
            total_count = sum(self.workload_counts.values())
            print(total_count)
            assert total_count == self.num_jobs

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
            # self.workload_weights is a dictionary
            keys = list(self.workload_weights.keys())
            weights = [self.workload_weights[key] for key in keys]
            return np.random.choice(keys, p=weights)
            # return np.random.choice(list(self.workload_catalog.keys()), p=self.workload_weights)
        
    def generate(self):
        task_id = 0
        trace = {} # job_id -> {}

        prev_job_arrival_time = 0
        self.set_seed(self.seed)
        if self.workload_counts is None:
            selected_workloads = [self.choose_workload() for _ in range(self.num_jobs)]
        else:
            selected_workloads = []
            for workload_name, count in self.workload_counts.items():
                selected_workloads.extend([workload_name] * count)
            random.shuffle(selected_workloads)

        self.set_seed(self.seed)
        self.set_seed(self.seed)
        durations = [int(10 ** random.uniform(1.5, 3) * 60) if random.random() <= 1 else int(10 ** random.uniform(3, 4) * 60) for _ in range(self.num_jobs)]
        # durations = [int(random.uniform(30, 180))*60 for _ in range(self.num_jobs)]
        job_arrival_times = np.cumsum(np.random.exponential(self.avail_time_lambda, self.num_jobs))
        # durations = [int(10 ** random.uniform(1.5, 2.5) * 60) for _ in range(self.num_jobs)]
        # while True:
        #     # the largest arrival + duration has to be < 36000
        #     job_arrival_times = np.cumsum(np.random.exponential(self.avail_time_lambda, self.num_jobs))
        #     # shift the arrival time, so that the first job arrives at 10
        #     job_arrival_times -= (job_arrival_times[0] - 10)
        #     durations = [int(random.uniform(30, 180))*60 for _ in range(self.num_jobs)]
        #     largest_end_time = max([job_arrival_times[job_id] + durations[job_id] for job_id in range(self.num_jobs)])
        #     if largest_end_time < 32000:
        #         break
        # durations = [int(random.uniform(30, 180))*60 for _ in range(self.num_jobs)]

        for job_id in range(self.num_jobs):
            # Generate job arrival time
            job_arrival_time = int(job_arrival_times[job_id])

            # Generate job
            workload_name = selected_workloads[job_id]
            workload = self.workload_catalog[workload_name]
            
            init_delay = workload.init_delay
            full_throughput = workload.workload_tasks[0].full_throughput
            support_throughput_aware = workload.support_throughput_aware

            # job duration
            duration = durations[job_id] if workload.duration is None else workload.duration
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
            # random select from 1, 2 ,4, 8
            num_tasks = random.choice([4])
            for i in range(num_tasks):
                workload_task = workload.workload_tasks[0]
                job_desc["tasks"][task_id] = {
                    "name": workload_task.name,
                    "demand": workload_task.demand,
                    "shm_size": workload_task.shm_size,
                    "image_id": workload_task.image_id,
                    "fetch_delay": workload_task.fetch_delay,
                    "build_image_delay": workload_task.build_image_delay,
                    "kill_delay": workload_task.kill_delay,
                    "upload_delay": workload_task.upload_delay
                }
                task_id += 1
            # for workload_task in workload.workload_tasks:
            #     job_desc["tasks"][task_id] = {
            #         "name": workload_task.name,
            #         "demand": workload_task.demand,
            #         "shm_size": workload_task.shm_size,
            #         "image_id": workload_task.image_id,
            #         "fetch_delay": workload_task.fetch_delay,
            #         "build_image_delay": workload_task.build_image_delay,
            #         "kill_delay": workload_task.kill_delay,
            #         "upload_delay": workload_task.upload_delay
            #     }
            #     task_id += 1

            trace[job_id] = job_desc
        
        # check that the arrival time of the last job is way after the start time
        # + duration of the ending_job_id

        # if self.ending_job_id is not None and \
        #     trace[self.num_jobs-1]["arrival_time"] < trace[self.ending_job_id]["arrival_time"] + trace[self.ending_job_id]["duration"] + 10000:
        #     raise ValueError("The arrival time of the last job is not way after the start time + duration of the ending_job_id")
        
        return trace
    
    def generate_to_file(self, output_path):
        trace = self.generate()
        # summary
        workload_count = {}
        for job_id, job_desc in trace.items():
            workload_name = job_desc["name"]
            workload_count[workload_name] = workload_count.get(workload_name, 0) + 1
        # print(workload_count)
        with open(output_path, 'w') as file:
            json.dump(trace, file, indent=4)

def main(seed=0):
    file_dir = pathlib.Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-jobs", type=int, default=100)
    parser.add_argument("--ending-job-id", type=int, default=29)
    parser.add_argument("--avail-time-lambda", type=int, default=1200)
    parser.add_argument("--workload-catalog-path", type=str, default=f"../workloads/workload_catalog.json")
    # parser.add_argument("--workload-weights", type=float, nargs='*')
    parser.add_argument("--output-path", type=str, default=f'{file_dir}/../resubmission_simulation_experiments/multi_task_synthetic/default_mix_{seed-10}.json')
    parser.add_argument("--seed", type=int, default=seed)

    args = parser.parse_args()
    trace_generator = TraceGenerator(
        num_jobs=args.num_jobs,
        ending_job_id=args.ending_job_id,
        avail_time_lambda=args.avail_time_lambda,
        workload_catalog_path=args.workload_catalog_path,
        random_migration_time=False,
        seed=args.seed,
        # workload_counts={
        #     "vit": 2,
        #     "resnet18": 3,
        #     "sage": 3,
        #     "cyclegan": 5,
        #     "gpt2": 1,
        #     "a3c": 4,
        #     "seq": 4,
        #     "gcn": 4,
        #     "openfoam": 6
        # }
        # workload_weights={
        #     "vit": 0.1,
        #     "resnet18": 0.1,
        #     "sage": 0.1,
        #     "cyclegan": 0.1,
        #     "gpt2": 0.1,
        #     "a3c": 0.12,
        #     "seq": 0.12,
        #     "gcn": 0.11,
        #     "openfoam": 0.15
        # }
        # workload_weights=args.workload_weights
    )
    trace_generator.generate_to_file(args.output_path)


if __name__ == "__main__":
    for i in range(10, 20):
        main(seed=i)

                

    