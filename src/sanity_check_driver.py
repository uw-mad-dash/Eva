import json
import time
import subprocess

EVA_IP_ADDR="172.31.17.248"
EVA_PORT=50422

class Job:
    def __init__(self, id, name, arrival_time, local_path):
        self.id = id
        self.name = name
        self.arrival_time = arrival_time
        self.local_path = local_path

def submit(job):
    global_working_dir = f"workspace/job_{job.name}"
    command = [
        "python", "eva_submit.py",
        "--eva-ip-addr", str(EVA_IP_ADDR),
        "--eva-port", str(EVA_PORT),
        "--local-working-dir", job.local_path,
        "--global-working-dir", global_working_dir
    ]
    subprocess.Popen(command)

root_dir = "workloads/"
interarrival_time = 5
job_names = []
# job_names += ["sage", "vit", "gpt2"]
job_names += ["cyclegan", "resnet18", "sage", "vit", "gpt2"]
job_names += ["a3c", "gcn", "seq", "openfoam"]

start_time = time.time()
for i, job_name in enumerate(job_names):
    time.sleep(interarrival_time)
    job_id = i
    job = Job(
        id=job_id,
        name=job_name,
        arrival_time=i * interarrival_time,
        local_path=f"{root_dir}/{job_name}"
    )
    submit(job)
    print(f"Job {job_name} arrived at {i * interarrival_time}")

