import json
import time
import subprocess
import os

EVA_IP_ADDR="172.31.17.248"
EVA_PORT=50422
trace = "/home/ubuntu/eurosys_artifacts_eval/physical_trace.json"

class Job:
    def __init__(self, id, name, arrival_time, local_path):
        self.id = id
        self.name = name
        self.arrival_time = arrival_time
        self.local_path = local_path

def submit(job):
    global_working_dir = f"workspace/job_{job.id}"
    command = [
        "python", "eva_submit.py",
        "--eva-ip-addr", str(EVA_IP_ADDR),
        "--eva-port", str(EVA_PORT),
        "--local-working-dir", job.local_path,
        "--global-working-dir", global_working_dir
    ]
    subprocess.Popen(command)


with open(trace, 'r') as file:
    trace_json = json.load(file)

jobs = {}
for job_id, job_desc in trace_json.items():
    jobs[int(job_id)] = Job(
        id=int(job_id),
        name=job_desc['name'],
        arrival_time=job_desc['arrival_time'],
        local_path=f"/home/ubuntu/eurosys_artifacts_eval/physical_jobs/{job_desc['name']}"
        # local_path=f"/home/ubuntu/resubmission_physical_jobs/{job_desc['name']}/job_{job_id}"
    )
    # make sure the path exist
    # if not, raise exception
    if not os.path.exists(jobs[int(job_id)].local_path):
        raise Exception(f"Path {jobs[int(job_id)].local_path} does not exist")


start_time = time.time()
while len(jobs) > 0:
    current_time = time.time() - start_time
    job_ids_to_remove = []
    for job_id, job in jobs.items():
        if job.arrival_time <= current_time:
            print(f"Job {job_id} arrived at {current_time}")
            submit(job)
            job_ids_to_remove.append(job_id)
            # submit job to the scheduler

    for job_id in job_ids_to_remove:
        del jobs[job_id]
    time.sleep(1)

