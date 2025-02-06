import numpy as np
import math
import copy
import time as time_module
import pickle

from .scheduler import Scheduler
from master.job import Job, JobStatus
from master.task import Task, TaskStatus
from master.instance import Instance, InstanceStatus
from cloud_provisioner.instance_type import InstanceType

class OwlScheduler(Scheduler):
    def __init__(self, contention_map_file, contention_map_value=0.9):
        super().__init__()
        self._logger.removeHandler(self._logging_handler)
        # set level
        # self._logger.setLevel("ERROR")
        if contention_map_file is None:
            self.contention_map = None
            self.contention_map_value = contention_map_value
        else:
            with open(contention_map_file, "rb") as f:
                self.contention_map = pickle.load(f)
    
    def get_task_normalized_throughputs(self, task_names):
        """
        task_names is a tuple of f"{job_name}_{task_name}"
        The tuple has at most 2 elements
        """
        assert len(task_names) <= 2
        assert len(task_names) > 0

        if len(task_names) == 1:
            return [1]
        
        if self.contention_map is None:
            return [self.contention_map_value] * 2

        # make sure the task_names are sorted
        key = tuple(sorted(task_names))
        assert key in self.contention_map
        return self.contention_map[key]
    
    def get_report(self):
        return {
        }
    
    def get_min_instance_type(self, task_ids, tasks, instance_types):
        """
        For each task, classify it into the it family that has the cheapest it
        that can accommodate the task.

        Returns 
        * task_to_min_it_map: dict of task_id -> it_id
        """
        task_to_min_it_map = {}
        for task_id in task_ids:
            task = tasks[task_id]
            min_it_id = None
            min_cost = None

            for it_id, it in instance_types.items():
                if it.family not in task.demand_dict:
                    continue
                if np.all(task.demand_dict[it.family] <= it.capacity) and \
                    (min_it_id is None or it.cost < min_cost):
                    min_it_id = it_id
                    min_cost = it.cost
            
            task_to_min_it_map[task_id] = min_it_id
        
        return task_to_min_it_map

    def get_alignment_score(self, demand_vector, capacity, full_capacity):
        # assuming demand <= capacity already
        score = 0
        for i in range(len(demand_vector)):
            if full_capacity[i] == 0:
                continue
            score += (demand_vector[i] / full_capacity[i]) * (capacity[i] / full_capacity[i])
        return score
    
    def get_config_cost(self, planned_config, instances, instance_types):
        cost = 0 
        for instance_id in planned_config:
            if type(instance_id) is tuple:
                it_id = instance_id[1]
            else:
                it_id = instances[instance_id].instance_type_id
            cost += instance_types[it_id].cost
        cost = round(cost, 2)
        return cost / 3600


    def get_opportunity_cost(self, task_ids, tasks, jobs, instance_types, task_to_min_it_map):
        """
        get throughput normalized opportunity cost of the tasks in task_ids.
        """
        total_opportunity_cost = 0
        if len(task_ids) == 0:
            return 0

        assert len(task_ids) <= 2
        task_names = tuple([f"{jobs[tasks[task_id].job_id].name.split('[')[0]}_{tasks[task_id].name}" for task_id in task_ids])
        task_names = tuple(sorted(task_names))
        task_ids = sorted(task_ids, key=lambda x: task_names.index(f"{jobs[tasks[x].job_id].name.split('[')[0]}_{tasks[x].name}"))
        tputs = self.get_task_normalized_throughputs(task_names)

        for i, task_id in enumerate(task_ids):
            opportunity_cost = instance_types[task_to_min_it_map[task_id]].cost * tputs[i]
            total_opportunity_cost += opportunity_cost

        return total_opportunity_cost

    def get_provision_saving(self, config, instances, instance_types, tasks, jobs, task_to_min_it_map, contention_map):
        """
        get per sec saving of the config
        """
        actual_cost = self.get_config_cost(config, instances, instance_types)
        opportunity_cost = 0
        for instance_id in config:
            task_ids = config[instance_id]
            opportunity_cost += self.get_opportunity_cost(task_ids, tasks, jobs, instance_types, task_to_min_it_map)
        
        opportunity_cost /= 3600
        
        return opportunity_cost - actual_cost

        
    def calculate_migration_cost(self, current_config, planned_config, jobs, tasks, instance_types):
        moved_jobs = set()
        for instance_id in planned_config:
            for task_id in planned_config[instance_id]:
                if instance_id not in current_config or task_id not in current_config[instance_id]:
                    moved_jobs.add(tasks[task_id].job_id)
        self._logger.debug(f"moved_jobs: {moved_jobs}")
        self._logger.debug(f"their migration costs: {[jobs[job_id].get_migration_cost(tasks) for job_id in moved_jobs]}")
        task_cost = sum([jobs[job_id].get_migration_cost(tasks) for job_id in moved_jobs])
        instance_cost = sum([instance_types[instance_id[1]].cost / 3600 * 300 for instance_id in planned_config if type(instance_id) is tuple])
        return task_cost + instance_cost
        # return sum([jobs[job_id].get_migration_cost(tasks) for job_id in moved_jobs])
    
    def owl(self, planned_config, task_ids, instances, instance_types, task_to_min_it_map, tasks, jobs, contention_map):
        # determine the set of tasks to be considered for reconfiguration
        # this include 1. tasks that are not assigned to any instance
        # 2. tasks on instances that are not worth it anymore (using the static contention map)
        initial_task_ids = copy.copy(task_ids)
        candidate_task_ids = []
        for task_id in task_ids:
            if tasks[task_id].instance_id is None:
                self._logger.debug(f"task_id {task_id} is not on any instance")
                candidate_task_ids.append(task_id)
        for instance_id in instances:
            # ignore instances that have jobs that are migrating
            if instance_id in planned_config:
                continue
            if len(instances[instance_id].committed_task_ids) >= 2: # should be just ==
                continue
            candidate_task_ids.extend(instances[instance_id].committed_task_ids)
        
        self._logger.debug(f"candidate_task_ids: {candidate_task_ids}")
         # for tasks that are not candidate_task_ids, add them to planned config
        for task_id in task_ids:
            if task_id not in candidate_task_ids:
                instance_id = tasks[task_id].committed_instance_id
                planned_config.setdefault(instance_id, []).append(task_id)
        
        # start owl
        # loop throughall possible combinations of tasks (at most 2) in descending opportunity cost / instance cost
        task_names = list(set([f"{jobs[tasks[task_id].job_id].name}_{tasks[task_id].name}" for task_id in task_ids]))
        task_name_to_task_id = {f"{jobs[tasks[task_id].job_id].name}_{tasks[task_id].name}": task_id for task_id in task_ids} # for each task_name, pick any one task_id 
        # generate all possible combinations of tasks using task_names, up to 2 tasks
        task_name_combinations = []
        for task_name in task_names:
            task_name_combinations.append((task_name,))
        for i in range(len(task_names)):
            for j in range(i, len(task_names)):
                task_name_combinations.append(tuple(sorted([task_names[i], task_names[j]])))
        # for each combination, determine their min_it_id
        task_combination_to_min_it_map = {}
        for task_name_combination in task_name_combinations:
            min_it_id = None
            min_cost = None
            for it_id, it in instance_types.items():
                total_demand = np.zeros(len(it.capacity))
                # make sure all tasks have demand in the it family
                if any([it.family not in tasks[task_name_to_task_id[task_name]].demand_dict for task_name in task_name_combination]):
                    continue
                for task_name in task_name_combination:
                    task_id = task_name_to_task_id[task_name]
                    total_demand += tasks[task_id].demand_dict[it.family]
                if np.all(total_demand <= it.capacity) and (min_it_id is None or it.cost < min_cost):
                    min_it_id = it_id
                    min_cost = it.cost
            
            if min_it_id is None:
                # no instance type can accommodate this task combination, skip
                continue
            
            task_combination_to_min_it_map[task_name_combination] = min_it_id
        
        # remove combinations that are not feasible
        task_name_combinations = [task_name_combination for task_name_combination in task_name_combinations if task_name_combination in task_combination_to_min_it_map]
        
        # sort task_name_combinations by descending opportunity cost / instance cost
        task_name_combinations = sorted(task_name_combinations, key=lambda x: self.get_opportunity_cost([task_name_to_task_id[task_name] for task_name in x], tasks, jobs, instance_types, task_to_min_it_map) / instance_types[task_combination_to_min_it_map[x]].cost, reverse=True)
        for task_name_combination in task_name_combinations:
            self._logger.debug(f"{task_name_combination}: {self.get_opportunity_cost([task_name_to_task_id[task_name] for task_name in task_name_combination], tasks, jobs, instance_types, task_to_min_it_map) / instance_types[task_combination_to_min_it_map[task_name_combination]].cost}")

        for task_name_combination in task_name_combinations:
            while True:
                # see if there are tasks in candidate_task_ids that are in this combination
                chosen_task_ids = []
                for task_name in task_name_combination:
                    for task_id in candidate_task_ids:
                        if f"{jobs[tasks[task_id].job_id].name}_{tasks[task_id].name}" == task_name and \
                            task_id not in chosen_task_ids:
                            chosen_task_ids.append(task_id)
                            break
                if len(chosen_task_ids) != len(task_name_combination):
                    # skip this combination
                    break

                # assign this combination of task to a new instance
                if len(chosen_task_ids) == 1 and tasks[chosen_task_ids[0]].committed_instance_id is not None and instances[tasks[chosen_task_ids[0]].committed_instance_id].instance_type_id == task_combination_to_min_it_map[task_name_combination]:
                    instance_id = tasks[chosen_task_ids[0]].committed_instance_id
                    planned_config[instance_id] = chosen_task_ids
                    self._logger.debug(f"task_ids {chosen_task_ids} assigned back to its original instance_id {instance_id}")
                else:
                    min_it_id = task_combination_to_min_it_map[task_name_combination]
                    instance_id = (self._get_next_instance_id(), min_it_id)
                    planned_config[instance_id] = chosen_task_ids
                    self._logger.debug(f"task_ids {chosen_task_ids} assigned to new instance_id {instance_id}")
                # remove the assigned tasks from candidate_task_ids
                for task_id in chosen_task_ids:
                    candidate_task_ids.remove(task_id)
                
                if len(candidate_task_ids) == 0:
                    break

        # check if all tasks are assigned
        assigned_task_ids = [task_id for instance_id in planned_config for task_id in planned_config[instance_id]]
        assert len(assigned_task_ids) == len(set(assigned_task_ids))
        assert all([task_id in assigned_task_ids for task_id in initial_task_ids])

        return planned_config
    

    def generate_planned_config(self, 
                                jobs, # dict of job_id -> job
                                tasks, # dict of task_id -> task
                                instances, # dict of instance_id -> instance
                                instance_types, # dict of it_id -> instancetype
                                contention_map, # NOT USED!!!
                                up_instance_ids, # set of instance_ids
                                unfinished_job_ids, # set of job_ids
                                current_config,
                                time,
                                event_occurred,
                                real_reconfig=True
                                ):  
        if not real_reconfig:
            # turn of logging
            self._logger.disabled = True
        else:
            self._logger.disabled = False
        start_time = time_module.time()
        is_global_reconfig = True
        
        instances = {instance_id: instances[instance_id] for instance_id in self._get_assignable_instance_ids(instances, up_instance_ids)}
        planned_config = self._create_initial_config(jobs, tasks, instances, unfinished_job_ids)
        # non_reconfigurable_task_ids = [task_id for instance_id in planned_config for task_id in planned_config[instance_id]]

        # only deal with jobs that are not finished and migratable
        # can assume these tasks are either in queue, or already on some instance
        if not real_reconfig:
            # there should only be a single job in jobs
            job_ids = list(jobs.keys())
            self._logger.debug(f"fake reconfig")
            self._logger.debug(f"job_ids: {job_ids}")
        else:
            job_ids = self._get_reconfigurable_job_ids(jobs, unfinished_job_ids)
            self._logger.debug(f"real reconfig")
        task_ids = [task_id for job_id in job_ids for task_id in jobs[job_id].task_ids]
        self._logger.info(f"reconfigurable task_ids: {task_ids}")
        if len(task_ids) == 0:
            return planned_config

        task_to_min_it_map = self.get_min_instance_type(
            [task_id for job_id in unfinished_job_ids for task_id in jobs[job_id].task_ids], tasks, instance_types)
        
        planned_config = self.owl(
            planned_config, task_ids, instances, instance_types, task_to_min_it_map, tasks, jobs, contention_map)

        assigned_task_ids = [task_id for instance_id in planned_config for task_id in planned_config[instance_id]]
        # make sure all tasks are assigned exactly once
        self._logger.debug(f"task_ids: {task_ids}")
        self._logger.debug(f"assigned_task_ids: {assigned_task_ids}")
        assert len(assigned_task_ids) == len(set(assigned_task_ids))
        # make sure all tasks are assigned
        assert all([task_id in assigned_task_ids for task_id in task_ids])
        # make sure demand <= supply
        self._check_config_feasibility(planned_config, instances, instance_types, tasks)

        self._logger.info(f"planned_config: {planned_config}")


        return planned_config

