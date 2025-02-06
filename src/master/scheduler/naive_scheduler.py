import numpy as np

from .scheduler import Scheduler
from master.job import Job, JobStatus
from master.task import Task, TaskStatus
from master.instance import Instance, InstanceStatus
from cloud_provisioner.instance_type import InstanceType

class NaiveScheduler(Scheduler):
    def __init__(self):
        super().__init__()
    
    def generate_planned_config(self, jobs, # dict of job_id -> Job
                                tasks, # dict of task_id -> Task
                                instances, # dict of instance_id -> Instance
                                instance_types, # dict of it_id -> InstanceType
                                contention_map, # not used
                                up_instance_ids, # set of instance_ids that are up
                                unfinished_job_ids, # set of job_ids that are not finished
                                current_config,
                                time,
                                event_occurred,
                                real_reconfig=True
                                ):
        if not event_occurred:
            return current_config
        planned_config = self._create_initial_config(jobs, tasks, instances, unfinished_job_ids)

        # only deal with jobs that are not finished and migratable
        # can assume these tasks are either in queue, or already on some instance
        job_ids = self._get_reconfigurable_job_ids(jobs, unfinished_job_ids)
        task_ids = [task_id for job_id in job_ids for task_id in jobs[job_id].task_ids]
        for task_id in task_ids:
            # if the task is already on an instance, don't move it
            if tasks[task_id].instance_id is not None:
                planned_config.setdefault(tasks[task_id].instance_id, []).append(task_id)
            else:
                # find the least cost instance type that can run it
                best_it_id = None
                for it_id in instance_types:
                    # check if demand <= capacity
                    it_family = instance_types[it_id].family
                    if it_family in tasks[task_id].demand_dict and \
                        np.all(tasks[task_id].demand_dict[it_family] <= instance_types[it_id].capacity):
                        if best_it_id is None or instance_types[it_id].cost < instance_types[best_it_id].cost:
                            best_it_id = it_id
                if best_it_id is None:
                    raise Exception("No instance type found for task %s" % task_id)

                planned_config.setdefault((self._get_next_instance_id(), best_it_id), []).append(task_id)

        return planned_config