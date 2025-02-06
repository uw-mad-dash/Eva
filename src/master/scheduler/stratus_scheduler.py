import numpy as np
import math
import copy

from .scheduler import Scheduler
from master.job import Job, JobStatus
from master.task import Task, TaskStatus
from master.instance import Instance, InstanceStatus
from cloud_provisioner.instance_type import InstanceType

NUM_RESOURCE_TYPES = 3

class StratusScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        # disable logging
        self._logger.disabled = True

    def _create_bins(self, jobs, tasks, instances):
        """
        Create bins for tasks and instances based on their runtime.
        """
        task_to_bin_map = {} # task_id -> bin_id
        instance_to_bin_map = {} # instance_id -> bin_id

        for task_id in tasks:
            duration = jobs[tasks[task_id].job_id].duration
            execution_time = jobs[tasks[task_id].job_id].get_execution_time()
            remaining_time = max(0, duration - execution_time)
            bin_id = math.floor(math.log2(remaining_time))
            task_to_bin_map[task_id] = bin_id
        
        for instance_id in instances:
            tasks_on_instance = instances[instance_id].committed_task_ids
            bin_id = max([task_to_bin_map[task_id] for task_id in tasks_on_instance])
            instance_to_bin_map[instance_id] = bin_id
        
        return task_to_bin_map, instance_to_bin_map

    def _get_task_remaining_time(self, task_id, jobs, tasks):
        duration = jobs[tasks[task_id].job_id].duration
        execution_time = jobs[tasks[task_id].job_id].get_execution_time()
        self._logger.debug(f"task_id {task_id}: duration {duration}, execution_time {execution_time}")
        return max(1, duration - execution_time)

    def _get_task_bin_id(self, task_id, jobs, tasks):
        return math.floor(math.log2(self._get_task_remaining_time(task_id, jobs, tasks)))

    def _get_instance_bin_id(self, task_ids, jobs, tasks):
        # task_ids: the tasks on the instance
        return max([0]+[self._get_task_bin_id(task_id, jobs, tasks) for task_id in task_ids])

    def _get_instance_remaining_time(self, task_ids, jobs, tasks):
        # task_ids: the tasks on the instance
        return max([0]+[self._get_task_remaining_time(task_id, jobs, tasks) for task_id in task_ids])

    def _get_instance_remaining_capacity(self, instance_id, task_ids, tasks, instances, instance_types):
        # task_ids: the tasks on the instance
        full_capacity = copy.deepcopy(instance_types[instances[instance_id].instance_type_id].capacity)
        it_family = instance_types[instances[instance_id].instance_type_id].family
        used_capacity = np.zeros(len(full_capacity))
        for task_id in task_ids:
            assert it_family in tasks[task_id].demand_dict
            used_capacity += tasks[task_id].demand_dict[it_family]
        capacity = full_capacity - used_capacity
        return capacity
        
    
    def _pack(self, planned_config, task_ids, jobs, tasks, instances, instance_types):
        """
        Pack tasks to existing instances.
        """
        # up-pack
        self._logger.debug("packing tasks to existing instances")
        self._logger.debug(f"tasks: {task_ids}")
        existing_instance_ids = [instance_id for instance_id in instances]
        assigned_task_ids = []
        # for each task_id in task_ids, try to pack it to an instance with the same bin_id
        for task_id in task_ids:
            # Phase 1.1: try to pack to same bin
            self._logger.debug(f"task_id {task_id}")
            task = tasks[task_id]
            task_bin_id = self._get_task_bin_id(task_id, jobs, tasks)
            task_remaining_time = self._get_task_remaining_time(task_id, jobs, tasks)

            best_equal_pack_instance_id = None
            best_equal_pack_remaining_time_diff = None

            best_up_pack_instance_id = None
            best_up_pack_capacity = None

            best_down_pack_instance_id = None
            best_down_pack_capacity = None

            for instance_id in existing_instance_ids:
                it_family = instance_types[instances[instance_id].instance_type_id].family
                if it_family not in task.demand_dict:
                    continue
                tasks_on_instance = instances[instance_id].committed_task_ids
                instance_bin_id = self._get_instance_bin_id(tasks_on_instance, jobs, tasks)
                if instance_bin_id == task_bin_id:
                    # instance = instances[instance_id]
                    capacity = self._get_instance_remaining_capacity(instance_id, tasks_on_instance, tasks, instances, instance_types)
                    if np.any(task.demand_dict[it_family] > capacity):
                        continue
                    instance_remaining_time = self._get_instance_remaining_time(tasks_on_instance, jobs, tasks)
                    remaining_time_diff = abs(instance_remaining_time - task_remaining_time)
                    if best_equal_pack_instance_id is None or remaining_time_diff < best_equal_pack_remaining_time_diff:
                        best_equal_pack_instance_id = instance_id
                        best_equal_pack_remaining_time_diff = remaining_time_diff
                elif instance_bin_id > task_bin_id:
                    # instance = instances[instance_id]
                    capacity = self._get_instance_remaining_capacity(instance_id, tasks_on_instance, tasks, instances, instance_types)
                    if np.any(task.demand_dict[it_family] > capacity):
                        continue
                    if best_up_pack_instance_id is None or np.sum(capacity) > np.sum(best_up_pack_capacity):
                        best_up_pack_instance_id = instance_id
                        best_up_pack_capacity = capacity
                elif instance_bin_id < task_bin_id:
                    # instance = instances[instance_id]
                    capacity = self._get_instance_remaining_capacity(instance_id, tasks_on_instance, tasks, instances, instance_types)
                    if np.any(task.demand_dict[it_family] > capacity):
                        continue
                    if best_down_pack_instance_id is None or np.sum(capacity) > np.sum(best_down_pack_capacity):
                        best_down_pack_instance_id = instance_id
                        best_down_pack_capacity = capacity
            
            if best_equal_pack_instance_id is not None:
                # assign task to best_instance_id
                if best_equal_pack_instance_id < 0:
                    planned_config_instance_id = (best_equal_pack_instance_id, instances[best_equal_pack_instance_id].instance_type_id)
                else:
                    planned_config_instance_id = best_equal_pack_instance_id
                instance_bin_id = self._get_instance_bin_id(instances[best_equal_pack_instance_id].committed_task_ids, jobs, tasks)
                self._logger.debug(f"Equal-pack: assigning task_id {task_id} (bin_id {task_bin_id}) to instance_id {best_equal_pack_instance_id} (bin_id {instance_bin_id})")
                planned_config.setdefault(planned_config_instance_id, []).append(task_id)
                self._update_states(tasks, instances, planned_config)
                assigned_task_ids.append(task_id)
                continue
            if best_up_pack_instance_id is not None:
                # assign task to best_instance_id
                if best_up_pack_instance_id < 0:
                    planned_config_instance_id = (best_up_pack_instance_id, instances[best_up_pack_instance_id].instance_type_id)
                else:
                    planned_config_instance_id = best_up_pack_instance_id
                instance_bin_id = self._get_instance_bin_id(instances[best_up_pack_instance_id].committed_task_ids, jobs, tasks)
                self._logger.debug(f"Up-pack: assigning task_id {task_id} (bin_id {task_bin_id}) to instance_id {best_up_pack_instance_id} (bin_id {instance_bin_id})")
                planned_config.setdefault(planned_config_instance_id, []).append(task_id)
                self._update_states(tasks, instances, planned_config)
                assigned_task_ids.append(task_id)
                continue
            if best_down_pack_instance_id is not None:
                # assign task to best_instance_id
                if best_down_pack_instance_id < 0:
                    planned_config_instance_id = (best_down_pack_instance_id, instances[best_down_pack_instance_id].instance_type_id)
                else:
                    planned_config_instance_id = best_down_pack_instance_id
                instance_bin_id = self._get_instance_bin_id(instances[best_down_pack_instance_id].committed_task_ids, jobs, tasks)
                self._logger.debug(f"Down-pack: assigning task_id {task_id} (bin_id {task_bin_id}) to instance_id {best_down_pack_instance_id} (bin_id {instance_bin_id})")
                planned_config.setdefault(planned_config_instance_id, []).append(task_id)
                self._update_states(tasks, instances, planned_config)
                assigned_task_ids.append(task_id)
                continue
        
        task_ids = [task_id for task_id in task_ids if task_id not in assigned_task_ids]
        self._logger.debug(f"task_ids after packing to existing instances: {task_ids}")

        return planned_config
    
    def _construct_candidate_groups(self, task_ids, jobs, tasks, instances, instance_types):
        # the i-th group has the first i tasks in descending runtime
        sorted_task_ids = sorted(task_ids, key=lambda task_id: jobs[tasks[task_id].job_id].duration, reverse=True)

        candidate_groups = []
        cur_num_task_in_group = 1
        i = 0
        while i < len(sorted_task_ids):
            total_demand = {it_family: np.zeros(NUM_RESOURCE_TYPES) for it_family in set([instance_types[it_id].family for it_id in instance_types])}
            cg = []
            for j in range(i, min(i + cur_num_task_in_group, len(sorted_task_ids))):
                # if adding this task will exceed the capacity of all instance types, stop
                can_be_accommodated = False
                for it_id in instance_types:
                    it_family = instance_types[it_id].family
                    if it_family in tasks[sorted_task_ids[j]].demand_dict and \
                        np.all(total_demand[it_family] + tasks[sorted_task_ids[j]].demand_dict[it_family] <= instance_types[it_id].capacity):
                        can_be_accommodated = True
                        break
                if not can_be_accommodated:
                    break
                for it_family in total_demand:
                    if it_family in tasks[sorted_task_ids[j]].demand_dict:
                        total_demand[it_family] += tasks[sorted_task_ids[j]].demand_dict[it_family]
                    else:
                        # give super large value to prevent it family from being selected
                        total_demand[it_family] += 1000000
                cg.append(sorted_task_ids[j])

            self._logger.debug(f"candidate_group: {cg} has total_demand: {total_demand}")
            candidate_groups.append(cg)
            i += len(cg)
            cur_num_task_in_group += 1
        
        return candidate_groups

    def _compute_efficiency_score(self, demand_vector, it_id, instance_types):
        # assuming total_demand <= capacity
        capacity = copy.deepcopy(instance_types[it_id].capacity)
        constraining_resource_type = None
        constraining_resource_ratio = 0
        for r in range(len(demand_vector)):
            if capacity[r] == 0:
                continue
            ratio = demand_vector[r] / capacity[r]
            if constraining_resource_type is None or ratio > constraining_resource_ratio:
                constraining_resource_type = r
                constraining_resource_ratio = ratio
        
        # find the capacity of the smallest / cheapest instance type in the same family
        it_ids = [it_id for it_id in instance_types if instance_types[it_id].family == instance_types[it_id].family]
        sorted_it_ids = sorted(it_ids, key=lambda it_id: instance_types[it_id].cost)
        min_it_capacity = instance_types[sorted_it_ids[0]].capacity

        normalized_used_constraining_resource = demand_vector[constraining_resource_type] / min_it_capacity[constraining_resource_type]

        return normalized_used_constraining_resource / instance_types[it_id].cost


    def _scale_out(self, planned_config, task_ids, jobs, tasks, instances, instance_types):
        """
        Scale out instances to accommodate the tasks in task_ids.
        """
        self._logger.debug("scaling out instances")
        # form candidate groups
        candidate_groups = self._construct_candidate_groups(task_ids, jobs, tasks, instances, instance_types)
        self._logger.debug(f"candidate_groups: {candidate_groups}")

        for cg in candidate_groups:
            best_it = None
            best_efficiency_score = None
            for it in instance_types:
                it_family = instance_types[it].family
                if any([it_family not in tasks[task_id].demand_dict for task_id in cg]):
                    continue
                total_demand = np.sum([tasks[task_id].demand_dict[it_family] for task_id in cg], axis=0)
                if np.any(total_demand > instance_types[it].capacity):
                    continue
                efficiency_score = self._compute_efficiency_score(total_demand, it, instance_types)
                if best_it is None or efficiency_score > best_efficiency_score:
                    best_it = it
                    best_efficiency_score = efficiency_score
            # provision a new instance with best_it
            instance_id = (self._get_next_instance_id(), best_it)
            planned_config[instance_id] = cg
            self._logger.debug(f"provisioning new instance {instance_id} with instance type {instance_types[best_it].name} for tasks {cg}")
        
        assigned_task_ids = [task_id for instance_id in planned_config for task_id in planned_config[instance_id]]
        assert all([task_id in assigned_task_ids for task_id in task_ids])

        return planned_config
    
    def _check_if_underutilized(self, instance_id, task_ids, tasks, instances, instance_types):
        if type(instance_id) == tuple:
            instance_id = instance_id[0]
        it_family = instance_types[instances[instance_id].instance_type_id].family
        total_demand = np.sum([tasks[task_id].demand_dict[it_family] for task_id in task_ids], axis=0)
        return np.all(total_demand < 0.5 * instance_types[instances[instance_id].instance_type_id].capacity)
                    
    def _update_states(self, tasks, instances, planned_config):
        for instance_id in planned_config:
            if type(instance_id) == tuple:
                instances[instance_id[0]] = Instance(
                    instance_id=instance_id[0],
                    instance_type_id=instance_id[1],
                    instance_type_name=None,
                    task_ids=None,
                    committed_task_ids=planned_config[instance_id],
                    status=InstanceStatus.RUNNING,
                    creation_time=None, ip_addr=None, public_ip_addr=None, 
                    ssh_user=None, ssh_key=None,
                    cloud_instance_id=None, get_timestamp_callback=None,
                    task_assignable=True
                )
            else:
                instances[instance_id].committed_task_ids = planned_config[instance_id]
            for task_id in planned_config[instance_id]:
                tasks[task_id].committed_instance_id = instance_id if type(instance_id) == int else instance_id[0]
        return tasks, instances

    def generate_planned_config(self, jobs, # dict of job_id -> Job
                                tasks, # dict of task_id -> Task
                                instances, # dict of instance_id -> Instance, will be modified
                                instance_types, # dict of it_id -> InstanceType
                                current_config,
                                time,
                                event_occurred, # not used
                                contention_map, # not used
                                up_instance_ids, # set of instance_ids that are up
                                unfinished_job_ids, # set of job_ids that are not finished
                                real_reconfig=True
                                ):
        # deep copy since stratus might modify these
        jobs = copy.deepcopy(jobs)
        tasks = copy.deepcopy(tasks)
        instances = copy.deepcopy(instances)

        instances = {instance_id: instance for instance_id, instance in instances.items() if instance_id in self._get_assignable_instance_ids(instances, up_instance_ids)}
        planned_config = self._create_initial_config(jobs, tasks, instances, unfinished_job_ids)
        self._logger.debug(f"initial planned_config: {planned_config}")

        instances_with_non_migratable_jobs = [instance_id for instance_id in planned_config]

        # only deal with jobs that are not finished and migratable
        # can assume these tasks are either in queue, or already on some instance
        job_ids = self._get_reconfigurable_job_ids(jobs, unfinished_job_ids)
        initial_task_ids = [task_id for job_id in job_ids for task_id in jobs[job_id].task_ids]
        task_ids = copy.copy(initial_task_ids)

        # if a task is already on an instance, don't move it
        for task_id in task_ids:
            if tasks[task_id].instance_id is not None:
                planned_config.setdefault(tasks[task_id].instance_id, []).append(task_id)
        
        # remove tasks that are already assigned
        assigned_task_ids = [task_id for instance_id in planned_config for task_id in planned_config[instance_id]]
        task_ids = [task_id for task_id in task_ids if task_id not in assigned_task_ids]

        # PACKING
        planned_config = self._pack(planned_config, task_ids, jobs, tasks, instances, instance_types)
        tasks, instances = self._update_states(tasks, instances, planned_config)
        assigned_task_ids = [task_id for instance_id in planned_config for task_id in planned_config[instance_id]]
        task_ids = [task_id for task_id in task_ids if task_id not in assigned_task_ids]

        # SCALE_OUT
        planned_config = self._scale_out(planned_config, task_ids, jobs, tasks, instances, instance_types)
        tasks, instances = self._update_states(tasks, instances, planned_config)

        assigned_task_ids = [task_id for instance_id in planned_config for task_id in planned_config[instance_id]]
        assert all([task_id in assigned_task_ids for task_id in initial_task_ids])


        for instance_id in list(planned_config.keys()):
            # only apply to old instances
            # if type(instance_id) == tuple:
            #     continue
            if instance_id in instances_with_non_migratable_jobs:
                continue

            if self._check_if_underutilized(instance_id, planned_config[instance_id], tasks, instances, instance_types):
                self._logger.debug(f"instance {instance_id} is underutilized")
                self._logger.debug(f"planned_config: {planned_config}")
                task_ids = copy.deepcopy(planned_config[instance_id])

                # if there's only one task, and this is the smallest instance type, don't remove it
                if len(task_ids) == 1:
                    # find the smallest instance type of this task
                    task_id = task_ids[0]
                    smallest_it_id = None
                    for it_id in instance_types:
                        it_family = instance_types[it_id].family
                        if it_family in tasks[task_id].demand_dict and \
                            np.all(tasks[task_id].demand_dict[it_family] <= instance_types[it_id].capacity):
                            if smallest_it_id is None or instance_types[it_id].cost < instance_types[smallest_it_id].cost:
                                smallest_it_id = it_id
                    if smallest_it_id == (instances[instance_id].instance_type_id if type(instance_id) == int else instance_id[1]):
                        continue
                self._logger.debug(f"flag")

                for task_id in task_ids:
                    tasks[task_id].committed_instance_id = None
                del instances[instance_id if type(instance_id) == int else instance_id[0]]
                del planned_config[instance_id]

                self._logger.debug(f"packing tasks {task_ids} to existing instances")
                planned_config = self._pack(planned_config, task_ids, jobs, tasks, instances, instance_types)
                tasks, instances = self._update_states(tasks, instances, planned_config)
                
                assigned_task_ids = [task_id for instance_id in planned_config for task_id in planned_config[instance_id]]
                task_ids = [task_id for task_id in task_ids if task_id not in assigned_task_ids]

                self._logger.debug(f"scaling out for tasks {task_ids}")
                planned_config = self._scale_out(planned_config, task_ids, jobs, tasks, instances, instance_types)
                tasks, instances = self._update_states(tasks, instances, planned_config)

        assigned_task_ids = [task_id for instance_id in planned_config for task_id in planned_config[instance_id]]
        # make sure all tasks are assigned exactly once
        self._logger.debug(f"initial_task_ids: {initial_task_ids}")
        self._logger.debug(f"assigned_task_ids: {assigned_task_ids}")
        assert len(assigned_task_ids) == len(set(assigned_task_ids))
        # make sure all tasks are assigned
        assert all([task_id in assigned_task_ids for task_id in initial_task_ids])

        # check if instances can host the tasks
        for instance_id in planned_config:
            if type(instance_id) == tuple:
                it_id = instance_id[1]
            else:
                it_id = instances[instance_id].instance_type_id
            it_family = instance_types[it_id].family
            tasks_on_instance = planned_config[instance_id]
            total_demand = np.sum([tasks[task_id].demand_dict[it_family] for task_id in tasks_on_instance], axis=0)
            assert np.all(total_demand <= instance_types[it_id].capacity)

        self._logger.debug(f"planned_config: {planned_config}")
        return planned_config

