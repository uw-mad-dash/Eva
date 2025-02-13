import numpy as np
import math
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import re
import hashlib
import os

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def get_color(name):
    # # Define a large set of colors
    # color_list = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS)

    # # Hash the name to get a number
    # hash_val = int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16)

    # # Use the hash to pick a color from the list
    # return color_list[hash_val % len(color_list)]
    hash_map = {
        "NaiveScheduler": "tab:blue",
        "JobPackScheduler": "tab:orange",
        "ClassifyGreedyReuseScheduler": "tab:green",
        "AsyncPeriodicScheduler": "tab:red",
    }

def get_max_time(report):
    max_time = 0
    for instance in report['instances'].values():
        for session in instance['active_session_queue']:
            end_time = session['shut_down_end_time'] or max_time  # Use current max_time if end_time is None
            max_time = max(max_time, end_time)
        for history in instance['history']:
            max_time = max(max_time, history['timestamp'])
    for job in report['jobs'].values():
        for history in job['history']:
            max_time = max(max_time, history['timestamp'])
    return int(math.ceil(max_time))

def is_instance_active(instance, time):
    for session in instance['active_session_queue']:
        if session['worker_register_start_time'] <= time <= (session['shut_down_start_time'] or time):
            return True
    return False

def get_job_arrival_time(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)
    job_arrival_time = {}
    for job_id, job in report['jobs'].items():
        job_arrival_time[job_id] = job['arrival_time']
    return job_arrival_time

def get_job_completion_time(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)
    job_completion_time = {}
    for job_id, job in report['jobs'].items():
        end_time = job['end_timestamp']
        if end_time is not None:
            job_completion_time[job_id] = end_time
    return job_completion_time

def summarize_instantaneous_cost(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)

    max_time = get_max_time(report)
    print(max_time)

    cost = {} # {time -> cost}
    for time in range(max_time + 1):
        cost[time] = 0
        for instance_id, instance in report['instances'].items():
            if is_instance_active(instance, time):
                it_id = instance['instance_type_id']
                cost[time] += report['instance_types'][str(it_id)]['cost'] / 3600  # Convert cost to per-second
    
    return cost

def plot_instantaneous_costs(scheduler_report_paths, out_path):
    fig, ax = plt.subplots(figsize=(14, 7))

    # Adjust space between plots
    plt.subplots_adjust(hspace=0.5)

    max_time = 0  # Initialize max_time to zero

    # Plotting each scheduler's utilization
    job_arrival_time_plotted = False

    for scheduler, report_path in scheduler_report_paths.items():
        print(scheduler)
        if not job_arrival_time_plotted:
            job_arrival_time = get_job_arrival_time(report_path)
            for job_id, arrival_time in job_arrival_time.items():
                ax.axvline(x=arrival_time, color='lightgray', linestyle='--', linewidth=0.7)

            job_completion_time = get_job_completion_time(report_path)
            for job_id, completion_time in job_completion_time.items():
                ax.axvline(x=completion_time, color='mistyrose', linestyle='--', linewidth=0.7)
            job_arrival_time_plotted = True

        report = json.load(open(report_path, 'r'))
        local_max_time = get_max_time(report)
        max_time = max(max_time, local_max_time)  # Update max_time to the maximum across all reports
        cost = summarize_instantaneous_cost(report_path)

        timesteps = np.arange(local_max_time + 1)

        sorted_cost = [cost[time] for time in range(local_max_time + 1)]
        ax.plot(timesteps, sorted_cost, label=scheduler, color=get_color(scheduler))
    
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Instantaneous Cost ($/s)')
    ax.set_title('Instantaneous Cost Over Time for Different Schedulers')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)  # Save the combined figure
    
def get_average_num_tasks_per_instance(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)

    max_time = get_max_time(report)
    print(max_time)
    num_resources = len(report['instance_types']['0']['capacity'])
    time_sample = range(0, max_time + 1, 300)

    # instance_is_up = {} # {time -> {instance_id -> is_up}}
    # for time in range(max_time + 1):
    #     instance_is_up[time] = {}
    #     for instance_id, instance in report['instances'].items():
    #         instance_is_up[time][instance_id] = is_instance_active(instance, time)

    # Initialize the full history for each timestamp and instance
    # {instance_id -> {time -> task_ids}}
    tasks_on_instance = {instance_id: {} for instance_id in report['instances'].keys()}
    # Populate the task history for each instance at each timestamp
    for instance_id, instance in report['instances'].items():
        for record in instance['history']:
            time = int(record["timestamp"])
            tasks_on_instance[instance_id][time] = record["task_ids"]
    
    # num_task_per_instance = {} # instance_id -> [num_tasks]
    # average_num_task_per_instance = {}
    # for instance_id in report['instances']:
    #     num_task_per_instance[instance_id] = []
    #     start_time = report['instances'][instance_id]['active_session_queue'][0]['running_start_time']
    #     for time in range(start_time, max_time + 1, 60):
    #         # print(report['instances'][instance_id]['history'])
    #         if is_instance_active(report['instances'][instance_id], time):
    #             record_time = max([t for t in tasks_on_instance[instance_id].keys() if t <= time])
    #             task_ids = tasks_on_instance[instance_id][record_time]
    #             if len(task_ids) == 0:
    #                 continue
    #             num_task_per_instance[instance_id].append(len(task_ids))
    #         elif len(num_task_per_instance[instance_id]) > 0:
    #             break
    #     if len(num_task_per_instance[instance_id]) == 0:
    #         print(f"Instance {instance_id} has no task")
    #         continue
    #     average_num_task_per_instance[instance_id] = np.max(num_task_per_instance[instance_id])
    #     # print(f"Instance {instance_id}: {average_num_task_per_instance[instance_id]}")
    
    # print("avg:", np.mean(list(average_num_task_per_instance.values())))
    # return average_num_task_per_instance

    num_task_per_instance = []
    for time in range(2592000, 5184000, 300):
        # count number of tasks active, instance active, and calculate average
        num_tasks_active = 0
        num_instances_active = 0
        for instance_id in report['instances']:
            if is_instance_active(report['instances'][instance_id], time):
                num_instances_active += 1
                record_time = max([t for t in tasks_on_instance[instance_id].keys() if t <= time])
                task_ids = tasks_on_instance[instance_id][record_time]
                num_tasks_active += len(task_ids)
        num_task_per_instance.append(num_tasks_active / num_instances_active if num_instances_active > 0 else 0)

    print(f"avg for {report_path}: {np.mean(num_task_per_instance)}")
    


def summarize_cluster_utilization(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)

    max_time = get_max_time(report)
    print(max_time)
    num_resources = len(report['instance_types']['0']['capacity'])
    time_sample = range(2592000, 5184000, 300)

    # instance_is_up = {} # {time -> {instance_id -> is_up}}
    # for time in range(max_time + 1):
    #     instance_is_up[time] = {}
    #     for instance_id, instance in report['instances'].items():
    #         instance_is_up[time][instance_id] = is_instance_active(instance, time)

    # Initialize the full history for each timestamp and instance
    # {instance_id -> {time -> task_ids}}
    tasks_on_instance = {instance_id: {} for instance_id in report['instances'].keys()}
    # Populate the task history for each instance at each timestamp
    for instance_id, instance in report['instances'].items():
        for record in instance['history']:
            time = int(record["timestamp"])
            tasks_on_instance[instance_id][time] = record["task_ids"]
        # for time in range(max_time + 1):
        #     if tasks_on_instance[time][instance_id] is None:
        #         if time == 0:
        #             tasks_on_instance[time][instance_id] = []  # Initialize the task history for the first timestamp
        #         else:
        #             tasks_on_instance[time][instance_id] = tasks_on_instance[time - 1][instance_id]

    # Calculate per instance utilization at each timestamp
    instance_utilization = {} # {time -> {instance_id -> {resource_type -> utilization}}
    cluster_utilization = {} # {time -> {resource_type -> utilization}}
    for time in time_sample:
        instance_utilization[time] = {}
        cluster_utilization[time] = {}
        cluster_capacities = {i: 0 for i in range(num_resources)}
        cluster_demands = {i: 0 for i in range(num_resources)}

        for instance_id in report['instances']:
            if is_instance_active(report['instances'][instance_id], time):
                # find the latest time that this instance has key in tasks_on_instance
                record_time = max([t for t in tasks_on_instance[instance_id].keys() if t <= time])
                task_ids = tasks_on_instance[instance_id][record_time]
                it_id = report['instances'][instance_id]['instance_type_id']
                family = report['instance_types'][str(it_id)]['family']
                capacity = np.array(report['instance_types'][str(it_id)]["capacity"])
                demands = [np.array(report['tasks'][str(task_id)]["demand_dict"][family]) for task_id in task_ids]
                instance_utilization[time][instance_id] = {}
                for resource_type in range(len(capacity)):
                    total_demand = np.sum([demand[resource_type] for demand in demands])
                    if total_demand > 0 and capacity[resource_type] == 0:
                        raise ValueError("Instance demand > 0 but capacity == 0. This should not happen")
                    total_utilization = total_demand / capacity[resource_type] if capacity[resource_type] > 0 else 0
                    instance_utilization[time][instance_id][resource_type] = total_utilization

                    cluster_capacities[resource_type] = cluster_capacities.get(resource_type, 0) + capacity[resource_type]
                    cluster_demands[resource_type] = cluster_demands.get(resource_type, 0) + total_demand
            else:
                # Instance is not active at this time
                instance_utilization[time][instance_id] = None
        
        for resource_type in range(len(cluster_capacities)):
            if cluster_demands[resource_type] > 0 and cluster_capacities[resource_type] == 0:
                raise ValueError("Cluster demand > 0 but capacity == 0. This should not happen")
            cluster_utilization[time][resource_type] = cluster_demands[resource_type] / cluster_capacities[resource_type] if cluster_capacities[resource_type] > 0 else None

    # for each resource type, print average cluster utilization
    for resource_type in range(num_resources):
        avg_cluster_utilization = np.mean([cluster_utilization[time][resource_type] for time in time_sample if cluster_utilization[time][resource_type] is not None])
        print(f"Average cluster utilization for resource type {resource_type}: {avg_cluster_utilization}")
    
    # replace None with 0
    for time in time_sample:
        for resource_type in range(num_resources):
            if cluster_utilization[time][resource_type] is None:
                cluster_utilization[time][resource_type] = 0

    return instance_utilization, cluster_utilization

def calculate_and_plot_utilization_for_single_scheduler(report_path, out_path):
    with open(report_path, 'r') as file:
        report = json.load(file)
    max_time = get_max_time(report)
    instance_utilization, cluster_utilization = summarize_cluster_utilization(report_path)

    timesteps = np.arange(max_time)
    instance_ids = sorted(report['instances'].keys())  # Sort instance IDs for consistent plotting
    stacked_utilization = {instance_id: np.zeros(max_time + 1) for instance_id in instance_ids}
    
    # Populate the stacked utilization for each instance at each time
    for time in range(max_time + 1):
        for instance_id in instance_ids:
            utilization = instance_utilization[time].get(instance_id)
            if utilization is not None:  # Instance is active
                stacked_utilization[instance_id][time] = utilization * 100  # Convert to percentage

    # Prepare data for stacked plot
    cumulative_data = np.row_stack([stacked_utilization[instance_id] for instance_id in instance_ids])

    # Plotting
    plt.figure(figsize=(14, 7))
    
    # Cumulative Utilization Plot
    plt.subplot(1, 2, 1)
    plt.stackplot(timesteps, cumulative_data, labels=[f'Instance {id_}' for id_ in instance_ids], alpha=0.8)
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Utilization (%)')
    plt.title('Cumulative Utilization Over Time')
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))  # Format y-axis as percentages
    plt.ylim(0, len(instance_ids) * 100)  # Set y-axis limits (0 to 100 * number of instances

    # Average Utilization Plot
    # average_utilization = np.sum(cumulative_data, axis=0) / (len(instance_ids) or 1)  # Avoid division by zero
    # plt.subplot(1, 2, 2)
    # plt.plot(timesteps, average_utilization, label="Average Utilization", color='r')
    # plt.xlabel('Time')
    # plt.ylabel('Average Utilization (%)')
    # plt.title('Average Utilization Over Time')
    # plt.legend()
    # plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))  # Format y-axis as percentages

    plt.subplot(1, 2, 2)
    plt.plot(timesteps, np.array(list(cluster_utilization.values())) * 100, label="Cluster Wide Utilization", color='b')  # Convert to percentage
    plt.xlabel('Time')
    plt.ylabel('Cluster Wide Utilization (%)')
    plt.title('Cluster Wide Utilization Over Time')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))  # Format y-axis as percentages

    plt.tight_layout()
    plt.savefig(out_path)  # Save the figure

def calculate_and_plot_utilization(scheduler_report_paths, out_path):
    num_resource_types = 3

    fig, axs = plt.subplots(num_resource_types, figsize=(14, 7), sharex=True)
    if num_resource_types == 1:
        axs = [axs]

    # Adjust space between plots
    plt.subplots_adjust(hspace=0.5)

    # plt.figure(figsize=(14, 7))
    max_time = 0  # Initialize max_time to zero

    # Plotting each scheduler's utilization
    job_arrival_time_plotted = False
    for scheduler, report_path in scheduler_report_paths.items():
        print(scheduler)
        report = json.load(open(report_path, 'r'))
        local_max_time = get_max_time(report)
        max_time = max(max_time, local_max_time)  # Update max_time to the maximum across all reports
        _, cluster_utilization = summarize_cluster_utilization(report_path)
        continue

        timesteps = np.arange(local_max_time + 1)

        resource_name = {0: "GPU", 1: "CPU", 2: "Mem"}

        for resource_type in range(num_resource_types):
            if not job_arrival_time_plotted:
                job_arrival_time = get_job_arrival_time(report_path)
                for job_id, arrival_time in job_arrival_time.items():
                    axs[resource_type].axvline(x=arrival_time, color='lightgray', linestyle='--', linewidth=0.7)
                job_completion_time = get_job_completion_time(report_path)
                for job_id, completion_time in job_completion_time.items():
                    axs[resource_type].axvline(x=completion_time, color='mistyrose', linestyle='--', linewidth=0.7)

            sorted_cluster_utilization = [cluster_utilization[time][resource_type] for time in range(local_max_time + 1)]
            axs[resource_type].plot(timesteps, np.array(sorted_cluster_utilization) * 100, label=scheduler, color=get_color(scheduler))
            axs[resource_type].set_ylabel(f'{resource_type} Utilization (%)')
            axs[resource_type].legend()
            axs[resource_type].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
            axs[resource_type].set_ylim([0, 100])  # Ensure y-axis is from 0 to 100%
            axs[resource_type].set_title(f'{resource_name[resource_type]} Utilization Over Time')

        job_arrival_time_plotted = True
        
        plt.savefig(out_path)  # Save the combined figure

        # plt.plot(timesteps, np.array(sorted_cluster_utilization) * 100, label=scheduler, color=get_color(scheduler))  # Convert to percentage
    
    plt.xlabel('Time')
    # plt.ylabel('Cluster Wide Utilization (%)')
    # plt.title('Cluster Wide Utilization Over Time for Different Schedulers')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))  # Format y-axis as percentages
    plt.tight_layout()
    plt.savefig(out_path)  # Save the combined figure

def calculate_useful_work_ratio(report_path):
    """
    only consider jobs that are completed
    """
    with open(report_path, 'r') as file:
        report = json.load(file)
    max_time = get_max_time(report)  # I assume you have this function defined elsewhere

    useful_work_ratios = {}
    for job_id, job in report['jobs'].items():
        if len(job["history"]) == 0 or job["history"][-1]["status"] != "FINISHED":
            continue
        makespan = job["end_timestamp"] - job["arrival_time"]
        execution_time = 0
        previous_execution_start_time = None
        for session in job["execution_session_queue"]:
            if previous_execution_start_time is not None:
                execution_time += session["migrating_start_time"] - previous_execution_start_time
            else:
                makespan -= session["execution_start_time"] - job["arrival_time"]
            previous_execution_start_time = session["execution_start_time"]
        # last session
        execution_time += job["end_timestamp"] - previous_execution_start_time if previous_execution_start_time is not None else 0

        useful_work_ratios[job_id] = 1 - (execution_time / makespan if makespan > 0 else 0)

    data_sorted = np.sort(list(useful_work_ratios.values()))
    yvals = np.arange(1, len(data_sorted)+1) / float(len(data_sorted))
    # print(useful_work_ratios)
    # print average
    print(f"{report_path}: Average useful work ratio: {np.mean(data_sorted)}")
    return data_sorted, yvals

def plot_useful_work_ratio(scheduler_report_paths, out_path):
    plt.figure(figsize=(8, 6))
    for scheduler, report_path in scheduler_report_paths.items():
        data_sorted, yvals = calculate_useful_work_ratio(report_path)
        plt.plot(data_sorted, yvals, marker='.', linestyle='-', color=get_color(scheduler), label=scheduler)
    plt.xlabel('Migration Overhead per Job')
    plt.ylabel('Cumulative Density')
    plt.title('CDF of Migration Overhead per Job for Different Schedulers')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.xlim(0, 1.0)
    plt.savefig(out_path)  # Save the figure

def calculate_migration_count(report_path):
    """
    only consider jobs that are completed
    """
    with open(report_path, 'r') as file:
        report = json.load(file)
    migration_counts = {}
    jct = {}
    for job_id, job in report['jobs'].items():
        if len(job["history"]) == 0 or job["history"][-1]["status"] != "FINISHED":
            continue
        migration_counts[job_id] = len(job["execution_session_queue"]) - 1
        jct[job_id] = job["end_timestamp"] - job["arrival_time"]

    data_sorted = np.sort(list(migration_counts.values()))
    yvals = np.arange(1, len(data_sorted)+1) / float(len(data_sorted))
    # print total number of migrations
    # print total number of jobs
    print(report_path)
    print(f"Total number of jobs: {len(data_sorted)}")
    print(f"Total number of migrations: {sum(data_sorted)}")
    print(f"Average number of migrations per job: {np.mean(data_sorted)}")
    if np.sum(list(migration_counts.values())) > 0:
        print(f"Average time between migration: {np.sum(list(jct.values())) / 3600 / np.sum(list(migration_counts.values()))}")
    else:
        print("No migrations")
    # print(f"Average time between migration for each job: {np.mean([jct[job_id] / max(1, migration_counts[job_id]) for job_id in migration_counts]) / 3600} hours")
    print(f"migration count per hour: {np.mean([migration_counts[job_id] / (jct[job_id] / 3600) for job_id in migration_counts])}")
    print(f"time between migration: {1/(np.mean([migration_counts[job_id] / (jct[job_id] / 3600) for job_id in migration_counts]))}")
    return data_sorted, yvals

def plot_migration_count(scheduler_report_paths, out_path):
    plt.figure(figsize=(8, 6))
    # randomly generate colors for each scheduler 
    for scheduler, report_path in scheduler_report_paths.items():
        data_sorted, yvals = calculate_migration_count(report_path)
        plt.plot(data_sorted, yvals, marker='.', linestyle='-', color=get_color(scheduler), label=scheduler)
    plt.xlabel('Migration Count per Job')
    plt.ylabel('Cumulative Density')
    plt.title('CDF of Migration Count for Different Schedulers')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)  # Save the figure

def summarize_instantaneous_configuration(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)

    max_time = get_max_time(report)
    print(max_time)

    config = {}
    for time in range(max_time + 1):
        config[time] = {} # {time -> {it_id -> count}}
        for instance_id, instance in report['instances'].items():
            if is_instance_active(instance, time):
                it_id = instance['instance_type_id']
                config[time][it_id] = config[time].get(it_id, 0) + 1
    
    # check p3 #vcpus <= 128
    # for time in range(max_time + 1):
    #     num_cpus = 0
    #     for it_id in config[time]:
    #         it_name = report['instance_types'][str(it_id)]['name']
    #         if it_name.startswith('p3'):
    #             num_cpus += report['instance_types'][str(it_id)]['capacity'][1] * config[time][it_id]
    #     if num_cpus > 128:
    #         print(f"Error: p3 #vcpus ({num_cpus}) > 128 at time {time}")
    
    # check r7 and c7 cpus <= 1152
    for time in range(max_time + 1):
        num_cpus = 0
        for it_id in config[time]:
            it_name = report['instance_types'][str(it_id)]['name']
            if it_name.startswith('r7') or it_name.startswith('c7'):
                num_cpus += report['instance_types'][str(it_id)]['capacity'][1] * config[time][it_id]
        if num_cpus > 1152:
            print(f"Error: r7/c7 #vcpus ({num_cpus}) > 1152 at time {time}")

    return config

def plot_instantaneous_configuration(scheduler_report_path, out_path):
    """
    Plot the configuration over time for a single scheduler
    """
    data = summarize_instantaneous_configuration(scheduler_report_path)

    with open(scheduler_report_path, 'r') as file:
        instance_types = json.load(file)['instance_types']

    # Enumerate colors for each instance type
    it_colors = {
        "p3.2xlarge": lighten_color("tab:blue", 0.3),
        "p3.8xlarge": lighten_color("tab:blue", 0.6),
        "p3.16xlarge": lighten_color("tab:blue", 1),
        "c7i.large": lighten_color("tab:green", 0.2),
        "c7i.xlarge": lighten_color("tab:green", 0.35),
        "c7i.2xlarge": lighten_color("tab:green", 0.5),
        "c7i.4xlarge": lighten_color("tab:green", 0.65),
        "c7i.8xlarge": lighten_color("tab:green", 0.8),
        "c7i.12xlarge": lighten_color("tab:green", 0.95),
        "c7i.16xlarge": lighten_color("tab:green", 1.1),
        "c7i.24xlarge": lighten_color("tab:green", 1.25),
        "c7i.48xlarge": lighten_color("tab:green", 1.5),
        "r7i.large": lighten_color("tab:orange", 0.2),
        "r7i.xlarge": lighten_color("tab:orange", 0.4),
        "r7i.2xlarge": lighten_color("tab:orange", 0.6),
        "r7i.4xlarge": lighten_color("tab:orange", 0.8),
        "r7i.8xlarge": lighten_color("tab:orange", 1),
        "r7i.12xlarge": lighten_color("tab:orange", 1.2),
        "r7i.16xlarge": lighten_color("tab:orange", 1.4),
        "r7i.24xlarge": lighten_color("tab:orange", 1.6),
        "r7i.48xlarge": lighten_color("tab:orange", 1.8),
    }

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot each instance type as a stacked line
    it_points = {it_name: [] for it_name in it_colors.keys()}
    for time, config in data.items():
        seen_it_names = set()
        for it_id, count in config.items():
            it_id = str(it_id)
            it_name = instance_types[it_id]['name']
            it_points[it_name].append(count)
            seen_it_names.add(it_name)
        for it_name in it_colors:
            if it_name not in seen_it_names:
                it_points[it_name].append(0)


    # plot each instance type as a line
    # plot as stack plot
    cumulative_data = np.row_stack([it_points[it_name] for it_name in it_colors])
    ax.stackplot(list(data.keys()), cumulative_data, labels=[it_name for it_name in it_colors], colors=[it_colors[it_name] for it_name in it_colors], alpha=0.8)

    job_arrival_time = get_job_arrival_time(scheduler_report_path)
    for job_id, arrival_time in job_arrival_time.items():
        ax.axvline(x=arrival_time, color='grey', linestyle='--', linewidth=0.7)
        # add text
        ax.text(arrival_time, 0, f"Job {job_id} arrive", rotation=90, verticalalignment='bottom', horizontalalignment='right')
    job_completion_time = get_job_completion_time(scheduler_report_path)
    for job_id, completion_time in job_completion_time.items():
        ax.axvline(x=completion_time, color='red', linestyle='--', linewidth=0.7)
        ax.text(completion_time, 0, f"Job {job_id} complete", rotation=90, verticalalignment='bottom', horizontalalignment='right')

    # for it_name, points in reversed(it_points.items()):
    #     print(it_name)
    #     ax.plot(list(data.keys()), points, color=it_colors[it_name], label=it_name, linewidth=2)

    # for time, config in data.items():
    #     total_instances = 0
    #     for it_name in it_colors:
    #         for it_id, count in config.items():
    #             it_id = str(it_id)
    #             if instance_types[it_id]['name'] == it_name:
    #                 ax.plot([time, time + 1], [total_instances + count, total_instances + count], color=it_colors[it_name], linewidth=2)
    #                 total_instances += count

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Instances')
    ax.set_title('Instantaneous Configuration Over Time')
    # y axis 0 ~ 12
    ax.set_ylim([0, 5])

    # Set legend based on instance colors
    handles = [plt.Line2D([0], [0], color=it_colors[it_name], label=it_name) for it_name in it_colors]
    labels = [it_name for it_name in it_colors]
    ax.legend(handles, labels)

    # place legend outsie the plot
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    # ax.legend(handles, labels)

    # Save the plot
    plt.tight_layout()
    plt.savefig(out_path)

def get_job_duration(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)
    job_durations = {}
    for job_id, job in report['jobs'].items():
        job_durations[job_id] = job["duration"]
    return job_durations

def get_jct(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)
    jct = {}
    for job_id, job in report['jobs'].items():
        if len(job["history"]) == 0 or job["history"][-1]["status"] != "FINISHED":
            continue
        # if job["duration"] < 60000:
        #     continue
        jct[job_id] = job["end_timestamp"] - job["arrival_time"]
    return jct

def get_execution_time(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)
    execution_times = {}
    for job_id, job in report['jobs'].items():
        if len(job["history"]) == 0 or job["history"][-1]["status"] != "FINISHED":
            continue
        execution_time = 0
        previous_execution_start_time = None
        for session in job["execution_session_queue"]:
            if previous_execution_start_time is not None:
                execution_time += session["migrating_start_time"] - previous_execution_start_time
            previous_execution_start_time = session["execution_start_time"]
        # last session
        execution_time += job["end_timestamp"] - previous_execution_start_time if previous_execution_start_time is not None else 0
        execution_times[job_id] = execution_time
    return execution_times

def calculate_execution_time_and_makespan(report_path):
    """
    only consider jobs that are completed
    """
    with open(report_path, 'r') as file:
        report = json.load(file)
    max_time = get_max_time(report)  # I assume you have this function defined elsewhere

    execution_times = {}
    makespans = {}
    for job_id, job in report['jobs'].items():
        if len(job["history"]) == 0 or job["history"][-1]["status"] != "FINISHED":
            continue
        makespan = job["end_timestamp"] - job["arrival_time"]
        execution_time = 0
        previous_execution_start_time = None
        for session in job["execution_session_queue"]:
            if previous_execution_start_time is not None:
                execution_time += session["migrating_start_time"] - previous_execution_start_time
            # else:
            #     makespan -= session["execution_start_time"] - job["arrival_time"]
            previous_execution_start_time = session["execution_start_time"]
        # last session
        execution_time += job["end_timestamp"] - previous_execution_start_time if previous_execution_start_time is not None else 0
        execution_times[job_id] = execution_time
        makespans[job_id] = makespan


    return execution_times, makespans

def plot_migration_overhead_vs_duration(scheduler_report_path, out_path):
    fig, ax = plt.subplots(figsize=(14, 7))

    # Adjust space between plots
    plt.subplots_adjust(hspace=0.5)

    report_path = scheduler_report_path
    scheduler = scheduler_report_path.split('/')[-1].split('_')[0]
    execution_times, makespans = calculate_execution_time_and_makespan(report_path)
    useful_ratios = [makespans[job_id] / execution_times[job_id] for job_id in execution_times.keys()]
    ax.scatter(list(execution_times.values()), list(useful_ratios), label=scheduler, color=get_color(scheduler))
    
    min_val = 0
    max_val = max(max(list(execution_times.values())), max(list(makespans.values())))

    # set limit
    # ax.set_xlim([min_val, max_val])
    # ax.set_ylim([min_val, max_val])

    # ax.set_xscale('log')
    # ax.set_yscale('log')

    # draw a y = x line
    # ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-')

    ax.set_xlabel('Execution Time')
    ax.set_ylabel('Makespan / Execution Time')
    # ax.set_title('Makespan vs Execution Time for ' + scheduler)
    # ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)  # Save the combined figure

    execution_time_range = [(1.5, 2), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4)]
    # for each range, plot a violin plot
    fig, ax = plt.subplots(figsize=(14, 7))
    violin_data = []
    for start, end in execution_time_range:
        data = [makespans[job_id]/execution_times[job_id] for job_id, execution_time in execution_times.items() if 60* 10**start <= execution_time < 60* 10**end]
        violin_data.append(data)
    ax.violinplot(violin_data, showmeans=True, showmedians=True)
    ax.set_xlabel('Execution Time')
    ax.set_ylabel('Makespan / Execution Time')
    # x axis labels
    ax.set_xticks(range(1, len(execution_time_range) + 1))
    ax.set_xticklabels([f"{start} - {end}" for start, end in execution_time_range])
    plt.tight_layout()
    plt.savefig(out_path.replace('.png', '_violin.png'))  # Save the combined figure

    # print stats
    print(f"{scheduler_report_path}: Average makespan-to-execution time ratio: {np.mean(useful_ratios)}")
    print(f"{scheduler_report_path}: Median makespan-to-execution time ratio: {np.median(useful_ratios)}")
    # 95
    print(f"{scheduler_report_path}: 95th percentile makespan-to-execution time ratio: {np.percentile(useful_ratios, 99)}")

def calculate_average_normalized_execution_time_and_makespan(scheduler_report_paths, base_scheduler):
    scheduler_execution_times = {}
    scheduler_makespans = {}
    for scheduler, report_path in scheduler_report_paths.items():
        execution_times, makespans = calculate_execution_time_and_makespan(report_path)
        scheduler_execution_times[scheduler] = execution_times
        scheduler_makespans[scheduler] = makespans
    
    base_execution_times = scheduler_execution_times[base_scheduler]
    base_makespans = scheduler_makespans[base_scheduler]

    results = {}

    for scheduler in scheduler_report_paths:
        execution_times = scheduler_execution_times[scheduler]
        makespans = scheduler_makespans[scheduler]
        normalized_execution_times = [execution_times[job_id] / base_execution_times[job_id] for job_id in execution_times.keys()]
        normalized_makespans = [makespans[job_id] / base_makespans[job_id] for job_id in makespans.keys()]
        print(f"{scheduler}: Average normalized execution time: {np.mean(normalized_execution_times)}")
        print(f"{scheduler}: Average normalized makespan: {np.mean(normalized_makespans)}")
        print("\n")
        results[scheduler] = {
            "execution_time": np.mean(normalized_execution_times),
            "makespan": np.mean(normalized_makespans)
        }
    
    return results


def calculate_total_cost(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)

    max_time = get_max_time(report)
    # print(max_time)
    total_cost = 0

    for time in range(max_time):
        for instance_id, instance in report['instances'].items():
            if is_instance_active(instance, time):
                # print(f"Instance {instance_id} is active at time {time}")
                it_id = instance['instance_type_id']
                total_cost += report['instance_types'][str(it_id)]['cost'] / 3600
        
    return total_cost

def calculate_total_cost_per_instance(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)

    max_time = get_max_time(report)
    final_cost = 0

    for instance_id, instance in report['instances'].items():
        total_cost = 0
        active_time = 0
        for time in range(max_time):
            if is_instance_active(instance, time):
                active_time += 1
                it_id = instance['instance_type_id']
                total_cost += report['instance_types'][str(it_id)]['cost'] / 3600
        print(f"Instance {instance_id} ({report['instance_types'][str(it_id)]['name']}): ${total_cost:.2f}, active time: {active_time} sec")
        final_cost += total_cost
    
    print(f"Total cost: ${final_cost:.2f}")

def summarize_instance_type_provisioned(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)
    
    # count the number of instances of each type provisioned
    instance_type_provisioned = {}
    for instance_id, instance in report['instances'].items():
        it_id = instance['instance_type_id']
        instance_type_provisioned[it_id] = instance_type_provisioned.get(it_id, 0) + 1
    
    # sort key
    for it_id in sorted(instance_type_provisioned.keys()):
        it_name = report['instance_types'][str(it_id)]['name']
        print(f"{it_name}: {instance_type_provisioned[it_id]}")
    print("\n")

def print_per_job_average_observed_throughput(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)
    
    max_time = get_max_time(report)
    job_throughput = {}
    job_names = {}
    for job_id, job in report['jobs'].items():
        job_throughput[job_id] = job["average_observed_throughput"]
        job_names[job_id] = job["name"]
    
    # compute average throughput per job name
    for job_name in set(job_names.values()):
        job_ids = [job_id for job_id in job_names if job_names[job_id] == job_name]
        total_throughput = sum([job_throughput[job_id] for job_id in job_ids])
        print(f"{job_name}: {total_throughput / len(job_ids)}")
    print("\n")
    
    # print sort by job names
    for job_id in sorted(job_throughput.keys(), key=lambda x: job_names[x]):
        print(f"{job_names[job_id]} ({job_id}): {job_throughput[job_id]}")
    
def extract_total_cost(log_file_path):
    """
    Extracts the total cost from the log file.

    Parameters:
    log_file_path (str): The path to the log file.

    Returns:
    float: The total cost if found, otherwise None.
    """
    # Initialize the total cost variable
    total_cost = None

    # Define a regex pattern to match the total cost line
    total_cost_pattern = re.compile(r'Total cost: ([\d.]+)')

    # Open and read the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            # Search for the total cost line using the regex pattern
            match = total_cost_pattern.search(line)
            if match:
                # Extract the total cost value
                total_cost = float(match.group(1))
                break

    return total_cost

def calculate_global_reconfig_ratio(report_path):
    with open(report_path, "r") as f:
        report = json.load(f)

    total_reconfig_count = len(report["scheduler"]["global_reconfig_or_not"])
    reconfig_count = report["scheduler"]["global_reconfig_or_not"].count(True)

    print(f"Reconfig count: {reconfig_count}")
    print(f"Total reconfig count: {total_reconfig_count}")
    print(f"Reconfig ratio: {reconfig_count / total_reconfig_count}")

    return reconfig_count / total_reconfig_count

def print_average_jct(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)

    jct = get_jct(report_path)
    print(f"{report_path}: Average JCT: {np.mean(list(jct.values()))}")

def print_average_normalized_throughput_all(all_report_paths):
    task_type_to_full_throughput = {}
    for scheduler, report_path in all_report_paths.items():
        with open(report_path, 'r') as file:
            report = json.load(file)
        for task_id, task in report['tasks'].items():
            job_id = str(task['job_id'])
            job_name = report['jobs'][job_id]['name']
            task_name = task['name']
            key = f"{job_name}_{task_name}"
            observed_throughputs = np.array(task['observed_throughputs'])
            # remove bottom and top 10%
            # observed_throughputs = np.sort(observed_throughputs)
            observed_throughputs = observed_throughputs[int(len(observed_throughputs) * 0.1):int(len(observed_throughputs) * 0.9)]
            if len(observed_throughputs) == 0:
                continue
            average_observed_throughput = np.mean(observed_throughputs)
            task_type_to_full_throughput[key] = task_type_to_full_throughput.get(key, []) + [average_observed_throughput]
    
    for key in task_type_to_full_throughput:
        task_type_to_full_throughput[key] = np.max(task_type_to_full_throughput[key])

    for scheduler, report_path in all_report_paths.items():
        with open(report_path, 'r') as file:
            report = json.load(file)

        task_id_to_normalized_throughput = {}
        job_seen = set()
        for task_id, task in report['tasks'].items():
            job_id = str(task['job_id'])
            job_name = report['jobs'][job_id]['name']
            key = f"{report['jobs'][job_id]['name']}_{task['name']}"
            # if task['job_id'] in job_seen:
            #     continue
            full_throughput = task_type_to_full_throughput[key]
            observed_throughputs = np.array(task['observed_throughputs'])
            # remove bottom and top 10%
            # observed_throughputs = np.sort(observed_throughputs)
            observed_throughputs = observed_throughputs[int(len(observed_throughputs) * 0.1):int(len(observed_throughputs) * 0.9)]
            if len(observed_throughputs) == 0:
                average_observed_throughput = 0
            else:
                average_observed_throughput = np.mean(observed_throughputs)
            # print(f"{task_id}: {average_observed_throughput} / {full_throughput} = {average_observed_throughput / full_throughput}")
            task_id_to_normalized_throughput[task_id] = average_observed_throughput / full_throughput
            job_seen.add(task['job_id'])
        
        # remove outliers
        print(f"{report_path}: Average normalized throughput: {np.mean(list(task_id_to_normalized_throughput.values()))}")

    
def print_average_normalized_throughput(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)

    task_id_to_normalized_throughput = {}
    task_type_to_full_throughput = {}
    for task_id, task in report['tasks'].items():
        job_id = str(task['job_id'])
        job_name = report['jobs'][job_id]['name']
        task_name = task['name']
        key = f"{job_name}_{task_name}"
        task_type_to_full_throughput[key] = task_type_to_full_throughput.get(key, []) + [task['average_observed_throughput']]
    
    for key in task_type_to_full_throughput:
        task_type_to_full_throughput[key] = np.max(task_type_to_full_throughput[key])

    for key in task_type_to_full_throughput:
        print(f"{key}: {task_type_to_full_throughput[key]}")

    for task_id, task in report['tasks'].items():
        job_id = str(task['job_id'])
        job_name = report['jobs'][job_id]['name']
        key = f"{report['jobs'][job_id]['name']}_{task['name']}"
        full_throughput = task_type_to_full_throughput[key]
        average_observed_throughput = task['average_observed_throughput']
        print(f"{task_id}: {average_observed_throughput} / {full_throughput} = {average_observed_throughput / full_throughput}")
        task_id_to_normalized_throughput[task_id] = average_observed_throughput / full_throughput
    
    print(f"{report_path}: Average normalized throughput: {np.mean(list(task_id_to_normalized_throughput.values()))}")

def print_average_normalized_throughput_naive(report_path, naive_path):
    with open(report_path, 'r') as file:
        report = json.load(file)

    with open(naive_path, 'r') as file:
        naive_report = json.load(file)

    task_id_to_normalized_throughput = {}
    task_type_to_full_throughput = {}
    for task_id, task in naive_report['tasks'].items():
        job_id = str(task['job_id'])
        job_name = report['jobs'][job_id]['name']
        task_name = task['name']
        key = f"{job_name}_{task_name}_{task_id}"
        task_type_to_full_throughput[key] = task_type_to_full_throughput.get(key, []) + [task['average_observed_throughput']]
    
    for key in task_type_to_full_throughput:
        task_type_to_full_throughput[key] = np.max(task_type_to_full_throughput[key])

    # for key in task_type_to_full_throughput:
    #     print(f"{key}: {task_type_to_full_throughput[key]}")

    task_id_to_normalized_throughput = {}
    job_seen = set()
    for task_id, task in report['tasks'].items():
        job_id = str(task['job_id'])
        job_name = report['jobs'][job_id]['name']
        key = f"{report['jobs'][job_id]['name']}_{task['name']}_{task_id}"
        # if task['job_id'] in job_seen:
        #     continue
        full_throughput = task_type_to_full_throughput[key]
        average_observed_throughput = task['average_observed_throughput']
        normalized_throughput = average_observed_throughput / full_throughput
        if normalized_throughput > 1:
            continue
        # print(f"{task_id}: {average_observed_throughput} / {full_throughput} = {average_observed_throughput / full_throughput}")
        task_id_to_normalized_throughput[task_id] = average_observed_throughput / full_throughput
        job_seen.add(task['job_id'])
    
    # remove outliers

    
    print(f"{report_path}: Average normalized throughput: {np.mean(list(task_id_to_normalized_throughput.values()))}")

def artifacts_eval():
    path = "/home/ubuntu/eva_report.json"
    plot_instantaneous_configuration(path, "instantaneous_configuration.png")
    calculate_total_cost_per_instance(path)

def main():
    artifacts_eval()
    return
    # path = "/home/ubuntu/eva_report.json"
    # paths = {
    #     "eva": "/home/ubuntu/resubmission_physical_experiment/eva/eva_report.json",
    #     "stratus": "/home/ubuntu/resubmission_physical_experiment/stratus/eva_report.json",
    #     "synergy": "/home/ubuntu/resubmission_physical_experiment/synergy/eva_report.json",
    #     "owl": "/home/ubuntu/resubmission_physical_experiment/owl/eva_report.json",
    #     "naive": "/home/ubuntu/resubmission_physical_experiment/naive/eva_report.json"
    # }
    print_average_normalized_throughput_all(paths)
    # for scheduler in paths:
    #     print_average_jct(paths[scheduler])
    return

    for scheduler, path in paths.items():
        print_average_normalized_throughput_naive(path, paths["naive"])
    #     print(calculate_total_cost(path))
    
    # plot_migration_count(paths, "/home/ubuntu/resubmission_physical_experiment/migration_count_all_schedulers.png")
    # calculate_and_plot_utilization(paths, "/home/ubuntu/resubmission_physical_experiment/utilization_all_schedulers.png")

    
    # print(calculate_total_cost(path))
    # print(print_average_normalized_throughput(path))
    # print_average_normalized_throughput_naive(path, "/home/ubuntu/resubmission_physical_experiment/naive/eva_report.json")
    return
    # plot_instantaneous_configuration(path, "/home/ubuntu/physical_experiment/results/eva/instantaneous_configuration.png")
    # return

    base_dir = f"/home/ubuntu/eva/src/tmp_pai/"
    schedulers = ["NaiveScheduler", "StratusScheduler", "SynergyScheduler", "OwlScheduler", "EVAGangScheduler"] #, "InstanceCandidateScheduler", "FamilyAgnosticScheduler"]

    # plot_instantaneous_configuration(f"/home/ubuntu/eva/src/experiments_small_scale/AsyncPeriodicBenefitScheduler_32jobs_cand_real_time/report.json", f"/home/ubuntu/eva/src/parse_report/instantaneous_configuration_32job.png")
    # return
    # plot_instantaneous_configuration(f"/home/ubuntu/eva/src/fidelity_experiment/physical/eva_report.json", f"/home/ubuntu/eva/src/parse_report/instantaneous_configuration.png")
    # print(calculate_total_cost(f"/home/ubuntu/eva/src/fidelity_experiment/physical/eva_report.json"))
    # return

    traces = [
        "tmp_pai_trace"
        # "cand"
        # "tmp_pai_trace"
        # "short_long_300jobs_1_contention",
        # "short_long_300jobs_0.95_contention",
        # "short_long_300jobs_0.9_contention",
        # "short_long_300jobs_0.85_contention",
        # "short_long_300jobs_0.8_contention",
    ]
    for trace in traces:
        scheduler_report_paths = {}
        for scheduler in schedulers:
            scheduler_report_paths[scheduler] = f"{base_dir}/{scheduler}_{trace}/report.json"

    #     print(trace)
    #     # make sure dir exists
    #     os.makedirs(f"{base_dir}/{trace}", exist_ok=True)
    
        for scheduler in schedulers:
            print_average_jct(scheduler_report_paths[scheduler])
        #     plot_makespan_vs_execution_time(scheduler_report_paths[scheduler], f"{base_dir}/{trace}/{scheduler}_makespan_vs_execution_time.png")
        plot_migration_count(scheduler_report_paths, f"{base_dir}/{trace}_analysis/migration_count_all_schedulers.png")
        # # plot_useful_work_ratio(scheduler_report_paths, f"{base_dir}/{trace}/useful_work_ratio_all_schedulers.png")
        # plot_instantaneous_costs(scheduler_report_paths, f"{base_dir}/{trace}/instantaneous_cost_all_schedulers.png")
        calculate_and_plot_utilization(scheduler_report_paths, f"{base_dir}/{trace}_analysis/utilization_all_schedulers.png")
        # calculate_average_normalized_execution_time_and_makespan(scheduler_report_paths, "NaiveScheduler")

    # scheduler_report_paths = {
    #     "naive_scheduler": "/home/ubuntu/physical_experiment/results/naive/eva_report.json",
    #     "stratus_scheduler": "/home/ubuntu/physical_experiment/results/stratus/eva_report.json",
    #     "eva_scheduler": "/home/ubuntu/physical_experiment/results/eva/eva_report.json",
    # }

    # print_per_job_average_observed_throughput(scheduler_report_paths["naive_scheduler"])
    # return

    # calculate_and_plot_utilization(scheduler_report_paths, "/home/ubuntu/eva/src/parse_report/utilization_all_schedulers.png")
    # plot_migration_count(scheduler_report_paths, "/home/ubuntu/eva/src/parse_report/migration_count_all_schedulers.png")
    # plot_useful_work_ratio(scheduler_report_paths, "/home/ubuntu/eva/src/parse_report/useful_work_ratio_all_schedulers.png")
    for scheduler in scheduler_report_paths:
        calculate_migration_count(scheduler_report_paths[scheduler])
    return
    #     summarize_instance_type_provisioned(scheduler_report_paths[scheduler])
    # calculate_and_plot_utilization(scheduler_report_paths, "/home/ubuntu/eva/src/parse_report/utilization_all_schedulers.png")
    # plot_instantaneous_costs(scheduler_report_paths, "/home/ubuntu/eva/src/parse_report/instantaneous_cost_all_schedulers.png")
    calculate_average_normalized_execution_time_and_makespan(scheduler_report_paths, "naive_scheduler")

    # plot_instantaneous_costs({"small": "small.json"}, "/home/ubuntu/eva/src/parse_report/instantaneous_cost_all_schedulers.png")

if __name__ == "__main__":
    main()