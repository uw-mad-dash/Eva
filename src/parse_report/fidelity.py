from utils import get_max_time
import matplotlib.pyplot as plt
import json

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
        # total_throughput = sum([job_throughput[job_id] for job_id in job_ids])
        # print(f"{job_name}: {total_throughput / len(job_ids)}")
        min_throughput = min([job_throughput[job_id] for job_id in job_ids])  
        print(f"{job_name}: {min_throughput}")
    print("\n")

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
        job_name = job["name"]
        identifier = job_id # change this if you want to use job_id
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
        execution_times[identifier] = execution_time
        makespans[identifier] = makespan
    
    return execution_times, makespans

if __name__ == "__main__":
    scheduler = "stratus"
    physical_report = f"/home/ubuntu/mount/physical_experiment/results/{scheduler}/eva_report.json"
    simulation_report = f"/home/ubuntu/mount/physical_experiment/simulation_results/{scheduler}_real_throughput/report.json"
    physical_data = calculate_execution_time_and_makespan(physical_report)
    simulation_data = calculate_execution_time_and_makespan(simulation_report)
    with open(physical_report, 'r') as file:
        report = json.load(file)

    # plot simulation vs physical for makespan for each job
    # x value simulation, y value physical
    print_per_job_average_observed_throughput(physical_report)
    print_per_job_average_observed_throughput(simulation_report)
    job_names = set([report['jobs'][job_id]['name'] for job_id in physical_data[1]])
    for job_name in job_names:
        for job_id in physical_data[1]:
            if report['jobs'][job_id]['name'] == job_name:
                print(f"Job {job_id} ({job_name}): Simulation Makespan: {simulation_data[1][job_id]}, Physical Makespan: {physical_data[1][job_id]}. {physical_data[1][job_id] / simulation_data[1][job_id]}")
                # break
    # for job_id in physical_data[1]:
    #     job_name = report['jobs'][job_id]['name']
    #     print(f"Job {job_id} ({job_name}): Simulation Makespan: {simulation_data[1][job_id]}, Physical Makespan: {physical_data[1][job_id]}")
    # match the jobs by their name
    data = {}
    for key in simulation_data[1].keys():
        if key in physical_data[1]:
            data[key] = (simulation_data[1][key], physical_data[1][key])
    plt.figure()
    plt.scatter([x[0] for x in data.values()], [x[1] for x in data.values()])
    # plt.plot([0, max([x[0] for x in data.values()])], [0, max([x[0] for x in data.values()])], color='r')
    # xlim 0, 11000, same for ylim
    x_min = 0
    x_max = 11000
    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
    # plot y = x line
    plt.plot([x_min, x_max], [x_min, x_max], color='r')

    plt.xlabel("Simulation JCT")
    plt.ylabel("Physical JCT")
    plt.title(f"Simulation vs Physical JCT ({scheduler})")
    plt.savefig("simulation_vs_physical_makespan.png")

    # print mse of percentage difference
    mse = 0
    for key, value in data.items():
        percentage_diff = (value[1] - value[0]) / value[1]
        print(key, percentage_diff)
        mse += percentage_diff ** 2
    mse /= len(data)
    # print sqrt of mse
    print("RMSE:", mse ** 0.5)

    


