class TaskStatus:
    SETTING_UP = 0
    RUNNING = 1
    KILLING = 2
    KILLED = 3
    EXITED = 4

class Task:
    def __init__(self, id, job_id, job_name, task_name, demand, shm_size, job_cloud_dir, task_relative_dir, 
        job_local_dir, task_local_dir,
        download_exclude_list, ip_address, bridge_ip_address, status):
        self.id = id
        self.job_id = job_id
        self.job_name = job_name
        self.task_name = task_name
        self.demand = demand
        self.shm_size = shm_size
        self.job_cloud_dir = job_cloud_dir
        self.task_relative_dir = task_relative_dir
        self.job_local_dir = job_local_dir
        self.task_local_dir = task_local_dir
        self.download_exclude_list = download_exclude_list
        self.ip_address = ip_address
        self.bridge_ip_address = bridge_ip_address
        self.status = status

        self.docker_image_name = f"task_{self.id}_{self.job_name}_{self.task_name}"
        self.docker_container_name = f"task_{self.id}_{self.job_name}_{self.task_name}"
        self.docker_container = None
        self.iterator_client = None

    ########################################################
    # status transition methods
    ########################################################
    def set_to_setting_up(self):
        self.status = TaskStatus.SETTING_UP
    
    def set_to_running(self):
        self.status = TaskStatus.RUNNING
    
    def set_to_killing(self):
        self.status = TaskStatus.KILLING
    
    def set_to_killed(self):
        self.status = TaskStatus.KILLED
    
    def set_to_exited(self):
        self.status = TaskStatus.EXITED
    
    def __str__(self):
        res = "Task("
        for key, value in self.__dict__.items():
            res += f"{key}={value}, "
        res += ")"
        return res