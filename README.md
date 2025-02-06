# EuroSys '25 Eva Artifacts
Eva is a system for cost-efficient hosting of batch jobs on the public cloud. This repository contains the artifacts for the EuroSys '25 submission.

## Eva Architecture
Eva consists of the following:
* Global Storage: A shared storage system that stores job artifacts provided by users and is accessible by all worker nodes.
* Master Node: The master node manages job submission and scheduling while also coordinating with the cloud provider to launch and shut down worker nodes.
* Worker Node: Each instance launched by the master node is a worker node. The worker node receives command from the master node to execute or terminate a job. To execute a job, the worker node fetches the job artifacts from the global storage. When the job is completed, the worker node syncs the job artifacts back to the global storage.

Current implementation of Eva uses Amazon S3 as global storage and Amazon EC2 as the cloud provider.

## Artifacts Evaluation
We describe three experiments for artifacts evaluation. Experiment 1 demonstrates Eva's functionality with a small-scale physical experiment. Experiment 2 compares Eva with baselines using simulation on partial Alibaba trace. Experiment 3 compares Eva with baselines using simulation on the full Alibaba trace.

Since deploying Eva requires access to AWS EC2 and S3, we provide a pre-configured c7i.xlarge EC2 instance with 4 vCPUs and 8 GB of memory for running Experiment 1. This instance serves as the master node and has the necessary permissions to launch additional instances for creating a small cloud-based cluster and accessing S3. It will be available throughout the artifact evaluation period. Experiment 2 and 3 can be conducted either on this EC2 instance or local machine. We also describe how to set up Eva on your own in the next section, but this is not necessary for the artifact evaluation.

### Experiment 1: Small Scale Physical Experiment 

We provide a pre-configured EC2 instance for running a small-scale experiment that demonstrates Eva’s functionality. The experiment should be about 20 minutes in total. The experiment includes three jobs:

* Job0: a two-node ResNet-18 training on a subset of ImageNet. Submitted at time 20 seconds. Duration around 7 minutes.
* Job1: a GraphSAGE model trained on OGBN-Arxiv. Submitted at time 30 seconds. Duration around 2 minutes.
* Job2: an A3C agent trained for Pong. Submitted at time 40 seconds. Duration around 4 minutes.

The jobs have resource demands shown in Table 6. The scheduler does periodic scheduling every 1 minute, so all three jobs will be present at the first scheduling period. The experiment should launch a total of 4 worker nodes in total: a `p3.8xlarge` for hosting Job0 and Job1, a `c7i.xlarge` for hosting Job2, and, once Job1 completes, 2 `p3.2xlarge` for hosting the two tasks of Job0. The instance will be terminated once jobs are migrated to new instances or completed. 

Follow the following steps to run the experiment on the pre-configured EC2 instance:

1. Contact the authors for the `eva_artifacts.pem` file. Place it at your home directory. Make sure the file has the correct permissions by running `chmod 400 eva_artifacts.pem`.
2. On your local machine, run `ssh -i "eva_artifacts.pem" ubuntu@ec2-54-234-34-211.compute-1.amazonaws.com`. The remaining commands should be run on the EC2 instance.
3. Run `cd eva/src`
4. Run `./run_artifact_evaluation.sh` to launch the experiment. This will launch a tmux session. The left panel is the scheduler in execution. The right panel is jobs being submitted.


### Experiment 2: Comparison with Baselines with Simulation on Partial Alibaba Trace
To demonstrate the functionality of the simulator, on the same EC2 instance we provide scripts to run a trace with 200 jobs from Alibaba trace. The experiment includes five schedulers: `No-Packing`, `Eva`, `Stratus`, `Owl`, and `Synergy`. The experiment should take about 20 minutes to complete.

Follow the following steps to run the experiment.

#### On the EC2 Instance
1. (If not done) Contact the authors for the `eva_artifacts.pem` file. Place it at your home directory.
2. (If not done)  On your local machine, run `ssh -i "eva_artifacts.pem" ubuntu@ec2-54-234-34-211.compute-1.amazonaws.com`. The remaining commands should be run on the EC2 instance.
3. Run `cd eva/src`
4. Run `python experiment_driver_200.py`. This will run the same trace with 5 different schedulers: No-Packing (`NaiveScheduler`), Eva (`EVAGangScheduler`), Stratus (`StratusScheduler`), Owl (`OwlScheduler`) and Synergy (`SynergyScheduler`).
5. After the simulation completes, the dollar cost will be printed. The full results will be saved in `simulation_expermients/<SCHEDULER>_pai_200`.

#### On Local Machine
1. Clone this repository.
2. `cd eva`
3. `pip install -r requirements.txt`
4. `cd src`
5. Run `python experiment_driver_200.py`. This will run the same trace with 5 different schedulers: No-Packing (`NaiveScheduler`), Eva (`EVAGangScheduler`), Stratus (`StratusScheduler`), Owl (`OwlScheduler`) and Synergy (`SynergyScheduler`).
6. After the simulation completes, the dollar cost will be printed. The full results will be saved in `simulation_expermients/<SCHEDULER>_pai_200`.

### Experiment 3: Simulation on Full Alibaba Trace
To reproduce the results in Table 9, one can run the simulation on the full Alibaba trace. The experiment takes about 6 hours to complete on a c7i EC2 instance. The actual time may vary depending on the local machine.

#### On the EC2 Instance
1. (If not done) Contact the authors for the `eva_artifacts.pem` file. Place it at your home directory.
2. (If not done)  On your local machine, run `ssh -i "eva_artifacts.pem" ubuntu@ec2-54-234-34-211.compute-1.amazonaws.com`. The remaining commands should be run on the EC2 instance.
3. Run `cd eva/src`
4. Run `python experiment_driver_full.py`. This will run the same trace with 5 different schedulers: No-Packing (`NaiveScheduler`), Eva (`EVAGangScheduler`), Stratus (`StratusScheduler`), Owl (`OwlScheduler`) and Synergy (`SynergyScheduler`).
5. After the simulation completes, the dollar cost will be printed. The full results will be saved in `simulation_expermients/<SCHEDULER>_pai_full`.

#### On Local Machine
1. Clone this repository.
2. `cd eva`
3. `pip install -r requirements.txt`
4. `cd src`
5. Run `python experiment_driver_full.py`. This will run the same trace with 5 different schedulers: No-Packing (`NaiveScheduler`), Eva (`EVAGangScheduler`), Stratus (`StratusScheduler`), Owl (`OwlScheduler`) and Synergy (`SynergyScheduler`).
6. After the simulation completes, the dollar cost will be printed. The full results will be saved in `simulation_expermients/<SCHEDULER>_pai_full`.


## Deploying Eva on Your Own
Currently, Eva is designed to run on AWS EC2 and use S3 as the global storage. To deploy Eva on AWS, follow the steps below.

### Eva Setup
#### Install Eva on All Nodes
Eventually, each node launched needs to have Eva installed. This can be done by preparing an AMI with Eva installed. To install Eva,
clone this repository and run `pip install -r requirements.txt` and `pip install -r ami_requirements.txt`.

In the future, we hope to make this step easier by making Eva a package that can be easily installed.

#### Update `eva_config.json`
This file contains the configuration for Eva. Update the following fields:
* Update `eva_ip_addr`, `master_ip_addr`, `swarm_ip_addr` to the IP address of the Eva master node.
* Update `bucket_name` to the name of the S3 bucket used. This will also be mounted on `~/mount` on every Eva worker node.

#### Update `ec2_config.json`
This file contains the configuration for EC2. Update the following fields:
* Update `region` to the region where you want to launch the instances.
* Update the content of `instance_types` to include the instance types that could be launched
* For each instance type, specify
    * `capacity`: list of `[GPU count, CPU count, Memory in GB]`
    * `cost`: cost per hour 
    * `family`: family of the instance type
    * `ssh_user`: the username to run on the instance
    * `launch_cfg`:
        * `key_name`: key name. This key should be present on the both master node and worker nodes (currently through storing on ami) to ensure access to EC2 and S3 resources.
        * `availability_zone`: default availability zone to launch the instance
        * `ami_id`: the AMI ID to launch the instance
        * `iam_role_name`: IAM role name. Make sure the role has access to S3 and EC2, and also passing role to the instance is enabled.
        * `instance_type`: name of the instance type
        * `instance_count`: number of instances to launch. Should be 1 in most cases.
        * `security_group`: list of security group name

#### Update `s3_config.json`
This file contains the configuration for S3. Update the bucket name.

### Jobs and Tasks
All the artifacts for the job should be placed in a directory. The directory should contain a `config.json` file that specifies the following:
* `name`: name of the job
* `tasks`: a dictionary of following
    * `name`: name of the task
    * `dir`: directory containing the task, relative to the job directory. This directory should contain a `Dockerfile`
    * `download_exclude_list`: list of files and directories to exclude from downloading from S3 under the same job directory. If left as empty list, all files will be downloaded.
    * `demand`: a dictionary specifying resource demand for each instance type family that this task can be launched on. The key is the family name (e.g. `p3`) and the demand is a list of `[GPU count, CPU count, Memory in GB]`. Key can also be `any` to specify the demand for any instance type.
    * `shm_size`: shared memory size in GB. This is used to set the shared memory size for the container.
    * `full_throughput`: the throughput of the task when running on a single instance without co-location.
* `support_throughput_aware`: whether or not this job has `eva_iterator` embedded in the code to track job throughput. If set to `false`, Eva will not track its throughput and thus will not be throughput and interference aware.
* `init_delay`: the delay in seconds from when the job gets executed to when the job actually makes progress. This is used to help scheduler decide job migration overhead.
* `fetch_delay`: the amount of time in seconds to download artifacts from S3. This is used to help scheduler decide job migration overhead.
* `build_image_delay`: the amount of time in seconds to build the docker image. This is used to help scheduler decide job migration overhead.
* `upload_delay`: the amount of time in seconds to upload artifacts to S3. This is used to help scheduler decide job migration overhead.
* `duration`: (needed only for Stratus) the duration of the job in seconds
* `kill_delay`: (needed only for simulation) the amount of time in seconds to kill the job.

Under the job directory (e.g. `two_node_resnet18/`), there could be multiple task subdirectories (e.g. `task0/`, `task1/`). This could also be just the same as the job directory in the case of single-task jobs (e.g. `a3c/` for job directory and `./` for task directory). The task directory should contain a `Dockerfile` that specifies the container image to be built and run the container. During task execution, the following environment variables are set:
* `NVIDIA_VISIBLE_DEVICES`: GPU IDs available to the task
* `CUDA_VISIBLE_DEVICES`: same as `NVIDIA_VISIBLE_DEVICES`
* `CPU_COUNT`: number of CPUs available to the task
* `WORKER_ID`: job-specific task ID (unique among all tasks within a job)
* `EVA_JOB_ID`: job ID
* `EVA_TASK_ID`: task ID (unique among all tasks across all jobs)
* `EVA_WORKER_IP_ADDR`: IP address of the worker node. This is used for `eva_iterator` to communicate with the worker node.
* `EVA_WORKER_PORT`: port number of the worker node. This is used for `eva_iterator` to communicate with the worker node.
* `EVA_ITERATOR_IP_ADDR`: IP address of the `eva_iterator` from the docker subnet.
* `EVA_ITERATOR_PORT`: port number of the `eva_iterator`.
* `EVA_START_TIMESTAMP`: start timestamp of the task in seconds since epoch. Used by `eva_iterator` to compute time elapsed.

Exemplar jobs can be found in `eva/src/workloads`.
### EVAIterator
To allow Eva to track task throughput and be interference aware, embed the `eva_iterator` in your batch job. The `eva_iterator` is a Python class that is used to track the throughput of the task. It can be easily wrapped around existing `DataLoader` and used as typical `DataLoader`:
```
dataloader = DataLoader(
    ImageDataset(opt.dataset_path, transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.num_workers,
)
dataloader = EVAIterator(dataloader)
```

### Execution
To launch the Eva master, run
```
python eva_master.py --config-path eva/src/eva_config.json
```
Once the master start running, users can submit their jobs by running
```
python eva_submit.py --eva-ip-addr <EVA_IP_ADDR> --eva-port <EVA_PORT> --local-working-dir <LOCAL_JOB_DIR> --global-working-dir <JOB_DIR_ON_GLOBAL_STORAGE>
```
If the job is already in global storage at `<JOB_DIR_ON_GLOBAL_STORAGE>`, user can omit `--local-working-dir`. After the job is finished, the master will sync the job artifacts back to `<JOB_DIR_ON_GLOBAL_STORAGE>`.



## Directory Structure
```
cloud_provisioner/              // code for cloud apis communicating with EC2. Can be modified to support other cloud providers.
eva_iterator/                   // code for custom iterator that is embedded in user code to track job throughput
master/                         // code for master
    scheduler/                  // code for scheduler. Can be modified to implement different scheduling policies.
pai_trace/                      // code for Alibaba PAI trace parsing and processing
    traces/                     // processed Alibaba PAI traces
parse_report/                   // code for parsing the experiment report
rpc/                            // code for rpc used in all components
simulation/                     // code for simulation, including generating trace, config used for simulation and measured contention map
simulator/                      // code for simulator
storage_manager/                // code for storage apis communicating with S3. Can be modified to support other storage providers.
worker/                         // code for worker
workloads/                      // code for jobs to be submitted to Eva

eva_master.py                   // entry script for master
eva_simulate.py                 // entry script for simulation
eva_submit.py                   // entry script for submitting jobs to Eva
eva_worker.py                   // entry script for worker
experiment_driver_200.py        // entry script for running simulation experiments on partial Alibaba trace (Experiment 2)
experiment_driver_full.py       // entry script for running simulation experiments on full Alibaba trace (Experiment 3)
job_submission_driver.py        // entry script for running automatic job submission in physical experiments
submission_manager.py           // submission interface used by Eva

run_artifact_evaluation.sh      // script for running the artifact evaluation small-scale experiment (Experiment 1)
run_eva_simulate.sh             // script for running eva_simulate.py
run_eva_submit.sh               // script for running eva_submit.py
run_physical.sh                 // script for running eva_master.py to start physical experiments

ec2_config.json                 // configuration file for EC2
eva_config.json                 // configuration file for Eva
s3_config.json                  // configuration file for S3
```
