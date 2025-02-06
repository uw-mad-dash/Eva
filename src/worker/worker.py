import argparse
import docker
import logging
import getpass
import multiprocessing
import shutil
import subprocess
import torch
import threading
import time
import ipaddress
from .task import Task, TaskStatus
from .custom_logging import WorkerAdapter
from .cpu import get_cpu_preference_list, get_num_physical_cores

from rpc.master_client import MasterClient
from rpc.worker_server import serve
from rpc.iterator_client import IteratorClient

LOG_FORMAT = "{name}:{lineno}:{levelname} {message}"

class Worker:
    def __init__(self, id, ip_addr, port, master_ip_addr, master_port, 
                 swarm_ip_addr, swarm_port, docker_network_name,
                 swarm_token, iterator_port, working_dir, datasets_dir, 
                 storage_manager_config,
                 mode, start_timestamp):
        self._id = id
        self._ip_addr = ip_addr
        self._port = port
        self._master_ip_addr = master_ip_addr
        self._master_port = master_port
        self._iterator_port = iterator_port
        self._mode = mode
        self._start_timestamp = start_timestamp

        self._storage_manager_config = storage_manager_config
        mod = __import__("storage_manager", fromlist=[storage_manager_config["class_name"]])
        storage_manager_class = getattr(mod, storage_manager_config["class_name"])
        self._storage_manager = storage_manager_class(**storage_manager_config["args"])

        self._lock = threading.Lock()

        self._tasks = {} # task_id -> task
        self._working_dir = working_dir
        self._datasets_dir = datasets_dir # root of the mounted dir that contains all datasets

        ##############################
        # Logging
        ##############################
        logger = logging.getLogger("worker")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, style="{"))
        logger.addHandler(handler)
        # also log to file
        file_handler = logging.FileHandler(f"/home/ubuntu/eva_worker_{self._id}.log")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, style="{"))
        logger.addHandler(file_handler)
        self._orig_logger = logger
        self._logger = WorkerAdapter(logger, {"worker": self})
        self._logging_handler = handler

        self._cpu_preference_list = get_cpu_preference_list()[:get_num_physical_cores()]
        self._cpu_allocation = {i: None for i in self._cpu_preference_list}
        self._gpu_allocation = {i: None for i in range(torch.cuda.device_count())}
        self._logger.info(f"CPU preference list: {self._cpu_preference_list}")
        # for now we don't need memory allocation because master 
        # will make sure that the memory used is within the limit

        callbacks = {
            "launch_task": self._launch_task_callback,
            "kill_task": self._kill_task_callback,
            "get_throughputs": self._get_throughputs_callback,
            "register_iterator": self._register_iterator_callback,
            "deregister_iterator": self._deregister_iterator_callback,
            "get_start_timestamp": self._get_start_timestamp_callback
        }

        self._server_thread = threading.Thread(
            target=serve,
            args=(self._ip_addr, self._port, callbacks)
        )
        self._server_thread.daemon = True
        self._server_thread.start()
        self._logger.info(f"Worker server started at {self._ip_addr}:{self._port}")
        
        self._docker_client = docker.from_env()
        # see if already in swarm
        if self._docker_client.info()["Swarm"]["LocalNodeState"]:
            self._logger.info("Already in swarm")
            # if deployed on ec2, this should not happen. We'll need to leave the swarm and join again
            if self._mode == "physical":
                self._docker_client.swarm.leave(force=True)
                self._docker_client.swarm.join(join_token=swarm_token, remote_addrs=[f"{swarm_ip_addr}:{swarm_port}"])
        else:
            self._docker_client.swarm.join(join_token=swarm_token, remote_addrs=[f"{swarm_ip_addr}:{swarm_port}"])
        
        self._docker_network_name = docker_network_name

        self._bridge_subnet = "192.0.0.0/16"
        self._bridge_iprange = "192.0.255.0/24"
        self._bridge_gateway = "192.0.255.1"
        self._bridge_network_name = "eva_bridge"
        self._bridge_network = self._create_bridge_network(self._bridge_network_name, self._bridge_subnet, self._bridge_iprange, self._bridge_gateway)
        self._bridge_network_free_ips = sorted(list(set(ipaddress.IPv4Network(self._bridge_subnet)) - set(ipaddress.IPv4Network(self._bridge_iprange))))



        self._master_client = MasterClient(
            self._master_ip_addr, self._master_port
        )
        self._master_client.RegisterWorker(self._id)

        self._monitor_thread = threading.Thread(target=self._monitor_tasks)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()


    ##############################
    # Private methods
    ##############################

    def _create_bridge_network(self, network_name, subnet, ip_range, gateway):
        """
        Create an bridge network with the given subnet and ip range
        """
        # check if the network already exists
        if network_name in [network.name for network in self._docker_client.networks.list()]:
            self._logger.info(f"Docker network {network_name} already exists")
            return self._docker_client.networks.get(network_name)
        
        try:
            ipam_config = docker.types.IPAMConfig(pool_configs=[
                docker.types.IPAMPool(subnet=subnet, iprange=ip_range, gateway=gateway)
            ])
            network = self._docker_client.networks.create(network_name, attachable=True, driver="bridge", ipam=ipam_config)
            self._logger.info(f"Created docker network {network_name} with subnet {subnet} and iprange {ip_range}")

            return network
        except docker.errors.APIError as e:
            self._logger.error(f"Failed to create docker network {network_name}: {e}")
            raise e

    def _get_free_bridge_ip_address(self):
        """
        Get a free ip address from the docker subnet
        """
        # assume lock is already acquired
        if len(self._bridge_network_free_ips) == 0:
            raise Exception("No free ip address available")

        ip = self._bridge_network_free_ips.pop(0)
        # if it is *.*.0.0, skip it
        if str(ip).endswith(".0"):
            ip = self._bridge_network_free_ips.pop(0)
        
        return str(ip)

    def _get_cpus(self, demand):
        # assume lock is already acquired
        cpus = []
        for i in self._cpu_preference_list:
            if self._cpu_allocation[i] is None:
                cpus.append(i)
            if len(cpus) == demand:
                break
        
        if len(cpus) < demand:
            raise Exception(f"Insufficient CPU resources")

        return cpus

    def _get_gpus(self, demand):
        # assume lock is already acquired
        gpus = []
        if demand == 0:
            return gpus
    
        for i in self._gpu_allocation:
            if self._gpu_allocation[i] is None:
                gpus.append(i)
            if len(gpus) == demand:
                break
        
        if len(gpus) < demand:
            raise Exception(f"Insufficient GPU resources")
        
        return gpus
    
    def _release_resources(self, task_id):
        # assume lock is already acquired
        self._logger.info(f"Releasing resources for task {task_id}")
        task = self._tasks[task_id]
        for cpu in self._cpu_allocation:
            if self._cpu_allocation[cpu] == task_id:
                self._cpu_allocation[cpu] = None
        for gpu in self._gpu_allocation:
            if self._gpu_allocation[gpu] == task_id:
                self._gpu_allocation[gpu] = None
    
    def task_cleanup(self, task_id):
        self._logger.info(f"Cleaning up task {task_id}")
        # task is already killed or exited
        task = self._tasks[task_id]
        # task.docker_container.remove() # keep it for debugging
        with self._lock:
            # task.docker_container = None
            self._logger.info(f"Acquired lock => to release resources for task {task_id}")
            self._release_resources(task_id)
        self._logger.info(f"Giving up lock => Released resources for task {task_id}")
        # write docker logs to a file
        log_file_name = f"task_{task_id}_{self.get_current_timestamp()}_instance_{self._id}.log"
        command = f"docker logs {task.docker_container_name} > {task.job_local_dir}/{log_file_name} 2>&1"
        self.run_subprocess(command)
        # subprocess.run(command, shell=True)

        self._logger.info(f"syncing back to cloud {task.job_local_dir} to {task.job_cloud_dir}")
        self._storage_manager.put_dir(
            src_path=task.job_local_dir,
            dst_path=task.job_cloud_dir
        )
        self._logger.info(f"Task {task_id} cleaned up")

        
    ##############################
    # Callbacks
    ##############################

    def _launch_task_callback(self, task_id, job_id, job_cloud_dir, task_relative_dir, download_exclude_list,
        demand, shm_size, ip_address, envs, job_name, task_name):
        """
        Launch the docker container for the task
        Returns fetch delay and build delay
        """
        try:
            self._logger.info(f"Callback: Launching task {task_id}")
            with self._lock:
                self._logger.info(f"Acquired lock => to launch task {task_id}")
                task = Task(
                    id=task_id,
                    job_id=job_id,
                    job_name=job_name,
                    task_name=task_name,
                    demand=demand,
                    shm_size=shm_size,
                    job_cloud_dir=job_cloud_dir,
                    task_relative_dir=task_relative_dir,
                    job_local_dir=f"{self._working_dir}/task_{task_id}_{job_name}_{task_name}/{job_cloud_dir}",
                    task_local_dir=f"{self._working_dir}/task_{task_id}_{job_name}_{task_name}/{job_cloud_dir}/{task_relative_dir}",
                    download_exclude_list=download_exclude_list,
                    ip_address=ip_address,
                    bridge_ip_address=self._get_free_bridge_ip_address(),
                    status=TaskStatus.SETTING_UP
                )
                self._tasks[task_id] = task
            self._logger.info(f"Giving up lock => Launched task {task_id}")
            self._logger.info(f"Added task: {task}")
            
            # move working_dir to local
            self._logger.info(f"Getting directory {task.job_cloud_dir} to {task.job_local_dir} with exclude list {download_exclude_list}")
            start_time = time.time()
            self._storage_manager.get_dir(
                src_path=task.job_cloud_dir,
                dst_path=task.job_local_dir,
                exclude_list=download_exclude_list)
            fetch_delay = int(time.time() - start_time)
            self._logger.info(f"Fetch delay: {fetch_delay}")

            # build image
            self._logger.info(f"Building image for task {task_id} at {task.task_local_dir}")
            start_time = time.time()
            # self._docker_client.images.build(
            #     path=task.task_local_dir,
            #     tag=task.docker_image_name
            # )
            command = f"docker build -t {task.docker_image_name} {task.task_local_dir}"
            # subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return_code = self.run_subprocess(command)
            if return_code != 0:
                raise Exception(f"Error building image for task {task_id}")
            build_delay = int(time.time() - start_time)
            self._logger.info(f"Build delay: {build_delay}")


            # launch container
            self._logger.info(f"Launching container for task {task_id}")
            with self._lock:
                self._logger.info(f"Acquired lock => to allocate rsc for task {task_id}")
                allocated_cpus = self._get_cpus(task.demand[1])
                for cpu in allocated_cpus:
                    self._cpu_allocation[cpu] = task_id
                self._logger.info(f"Allocated CPUs: {allocated_cpus} to task {task_id}")

                allocated_gpus = self._get_gpus(task.demand[0])
                for gpu in allocated_gpus:
                    self._gpu_allocation[gpu] = task_id
                self._logger.info(f"Allocated GPUs: {allocated_gpus} to task {task_id}")
            self._logger.info(f"Giving up lock => Allocated resources for task {task_id}")

            if task.demand[0] > 0:
                # run nvidia-smi to hopefully remind the instance that it has GPUs >:(
                command = "nvidia-smi"
                return_code = self.run_subprocess(command)
                if return_code != 0:
                    raise Exception(f"Error running nvidia-smi")

            cpuset_cpus = ",".join([str(i) for i in allocated_cpus])
            mem_limit = f"{task.demand[2]}g"
            shm_size = f"{task.shm_size}g"

            command = f"docker rm {task.docker_container_name}; docker container create \
                --privileged \
                --name {task.docker_container_name} \
                --network {self._bridge_network_name} \
                --ip {task.bridge_ip_address} \
                --cpuset-cpus {cpuset_cpus} \
                --memory {mem_limit} \
                --shm-size {shm_size} \
                --volume {task.job_local_dir}:/home:rw \
                --mount type=bind,source={self._datasets_dir},target=/datasets,readonly"
            
            if len(allocated_gpus) > 0:
                # command += f" --gpus {','.join([str(i) for i in allocated_gpus])}"
                command += f" --gpus '\"device={','.join([str(i) for i in allocated_gpus])}\"'"
                command += f" --runtime=nvidia"
                command += f" --env NVIDIA_VISIBLE_DEVICES={','.join([str(i) for i in allocated_gpus])}"
                command += f" --env CUDA_VISIBLE_DEVICES={','.join([str(i) for i in allocated_gpus])}"

            for k, v in envs.items():
                command += f" --env {k}={v}"
                
            command += f" --env CPU_COUNT={len(allocated_cpus)}"
            command += f" --env EVA_JOB_ID={job_id}"
            command += f" --env EVA_TASK_ID={task_id}"
            command += f" --env EVA_WORKER_IP_ADDR={self._ip_addr}"
            command += f" --env EVA_WORKER_PORT={self._port}"
            command += f" --env EVA_ITERATOR_IP_ADDR={task.bridge_ip_address}"
            command += f" --env EVA_ITERATOR_PORT={self._iterator_port}"
            command += f" --env EVA_START_TIMESTAMP={self._start_timestamp}"
            
            command += f" {task.docker_image_name}; "
            command += f"docker network connect {self._docker_network_name} --ip {task.ip_address} {task.docker_container_name}; "
            command += f"docker start {task.docker_container_name}"

            # self._logger.info(f"Running command: {command}")
            # subprocess.run(command, shell=True)
            return_code = self.run_subprocess(command, timeout=-1)
            if return_code != 0:
                raise Exception(f"Error launching container for task {task_id}")

            with self._lock:
                self._logger.info(f"Acquired lock => to get container for {task_id}")
                task.docker_container = self._docker_client.containers.get(task.docker_container_name)
                task.set_to_running()
                self._logger.info(f"Container for task {task_id} launched {task.docker_container.id}")
            self._logger.info(f"Giving up lock => Got container for {task_id}")

            self._logger.info(f"Task {task_id} launched")
            return True, fetch_delay, build_delay

        except Exception as e:
            self._logger.error(f"Error launching task: {e}")
            return False, 0, 0

    
    def _kill_task_callback(self, task_id):
        try:
            self._logger.info(f"Callback: Killing task {task_id}")
            if task_id not in self._tasks or self._tasks[task_id].status == TaskStatus.KILLED:
                self._logger.info(f"Task {task_id} already killed. Not killing again.")
                # the case where, the task completes while master is scheduling
                return True, -1
            
            task = self._tasks[task_id]
            with self._lock:
                self._logger.info(f"Acquired lock => to set task {task_id} to killing")
                task.set_to_killing()
            self._logger.info(f"Giving up lock => Set task {task_id} to killing")
            
            with self._lock:
                self._logger.info(f"Acquired lock => to reload container for {task_id}")
                task.docker_container.reload()
            
            # notify save checkpoint
            if task.iterator_client is not None:
                self._logger.info(f"Notifying save checkpoint for task {task_id}")
                success = task.iterator_client.NotifySaveCheckpoint()
                if not success:
                    raise Exception(f"Error notifying save checkpoint for task {task_id}")
                self._logger.info(f"Task {task_id} notified save checkpoint")

            while task.docker_container.status != "exited":
                self._logger.info(f"Task {task_id} has status {task.docker_container.status}. Gonna kill it.")
                try:
                    task.docker_container.kill()
                except Exception as e:
                    self._logger.error(f"Error killing container for task {task_id}: {e}")
                    self._logger.error(f"Retrying in 1 second")
                time.sleep(1)
                with self._lock:
                    self._logger.info(f"Acquired lock => to reload container for {task_id}")
                    task.docker_container.reload()
                self._logger.info(f"Giving up lock => Reloaded container for {task_id}")

            with self._lock:
                self._logger.info(f"Acquired lock => to set task {task_id} to killed")
                task.set_to_killed()
            self._logger.info(f"Giving up lock => Set task {task_id} to killed")

            start_time = time.time()
            self.task_cleanup(task_id)
            upload_delay = int(time.time() - start_time)

            self._logger.info(f"Task {task_id} killed")
            return True, upload_delay
        except Exception as e:
            self._logger.error(f"Error killing task {task_id}: {e}")
            return False, 0
    
    def _get_throughputs_callback(self):
        throughputs = {}
        # # TODO: remove for testing
        # return True, throughputs
        try:
            for task_id in self._tasks:
                task = self._tasks[task_id]
                if task.status == TaskStatus.RUNNING and task.iterator_client is not None:
                    success, ready, throughput = task.iterator_client.GetThroughput()
                    if not success:
                        raise Exception(f"Error getting throughput for task {task_id}")
                    if not ready:
                        self._logger.info(f"Task {task_id} is not ready for throughput.")
                        continue
                    throughputs[task_id] = throughput

            self._logger.info(f"Got throughputs: {throughputs}")
            return True, throughputs
        except Exception as e:
            self._logger.error(f"Error getting throughputs: {e}")
            return False, throughputs


    def _register_iterator_callback(self, task_id):
        try:
            with self._lock:
                self._logger.info(f"Acquired lock => to register iterator for task {task_id}")
                self._tasks[task_id].iterator_client = IteratorClient(
                    self._tasks[task_id].bridge_ip_address, self._iterator_port
                )
            self._logger.info(f"Giving up lock => Registered iterator for task {task_id}")
            return True
        except Exception as e:
            self._logger.error(f"Error registering iterator for task {task_id}: {e}")
            return False
    
    def _deregister_iterator_callback(self, task_id):
        try:
            with self._lock:
                self._logger.info(f"Acquired lock => to deregister iterator for task {task_id}")
                self._tasks[task_id].iterator_client = None
            self._logger.info(f"Giving up lock => Deregistered iterator for task {task_id}")
            return True
        except Exception as e:
            self._logger.error(f"Error deregistering iterator for task {task_id}: {e}")
            return False
    
    def _get_start_timestamp_callback(self):
        self._logger.info(f"Callback: Getting start timestamp")
        return True, self._start_timestamp
    
    ##############################
    # Serving threads
    ##############################

    def _monitor_tasks(self):
        while True:
            # task_to_remove = []
            container_statuses = {} # status => [task_ids]
            task_ids = list(self._tasks.keys())
            for task_id in task_ids:
                try:
                    task = self._tasks[task_id]
                    if task.docker_container is None:
                        continue
                    with self._lock:
                        self._logger.info(f"Acquired lock => to reload container for {task_id}")
                        task.docker_container.reload()
                    self._logger.info(f"Giving up lock => Reloaded container for {task_id}")

                    container_status = task.docker_container.status
                    container_statuses[container_status] = container_statuses.get(container_status, []) + [task_id]
                    if container_status == "exited" and task.status == TaskStatus.RUNNING:
                        self._logger.info(f"Task {task_id} exited")
                        with self._lock:
                            self._logger.info(f"Acquired lock => to set task {task_id} to exited")
                            task.set_to_exited()
                        self._logger.info(f"Giving up lock => Set task {task_id} to exited")
                        self.task_cleanup(task_id)
                        task.end_timestamp = self.get_current_timestamp()
                        self._master_client.TaskCompletion(self._id, task_id)
                except Exception as e:
                    self._logger.error(f"Error: {e}")

            # for task_id in task_to_remove:
            #     del self._tasks[task_id]
            self._logger.info(f"Container statuses: {container_statuses}")

            time.sleep(3)

    ##############################
    # Public methods
    ##############################

    def get_current_timestamp(self):
        return time.time() - self._start_timestamp
    
    def log_subprocess_output(self, pipe):
        res = ""
        for line in iter(pipe.readline, b''):
            res += line.decode("utf-8").strip()
        self._logger.info(res)
        # for line in iter(pipe.readline, b''):
        #     self._logger.info(line.decode("utf-8").strip())
    
    def run_subprocess(self, command, max_retry=3, timeout=300):
        self._logger.info(f"Running command: {command}")
        for i in range(max_retry):
            try:
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if timeout == -1:
                    process.wait()
                else:
                    process.wait(timeout=timeout)
                with process.stdout:
                    self.log_subprocess_output(process.stdout)
                return_code = process.returncode
                self._logger.info(f"Command {command} returned {return_code}")
                return return_code
            except subprocess.TimeoutExpired:
                # kill the process
                process.kill()
                self._logger.error(f"Command {command} timed out. Retrying {i+1}/{max_retry}")
                
        # process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # process.wait(timeout=300)
        # with process.stdout:
        #     self.log_subprocess_output(process.stdout)
        # return_code = process.returncode
        # self._logger.info(f"Command {command} returned {return_code}")
        # return return_code
        

    def send_heartbeat(self, worker_id):
        # TODO: implement this
        pass



