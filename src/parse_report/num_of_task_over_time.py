import json
import matplotlib.pyplot as plt

# Read JSON data from file
with open('../simulation/config/short_long_rand_checkpoint.json', 'r') as file:
    data = json.load(file)

# Create lists to store the arrival and departure times of tasks and jobs
events = []

# Iterate through each job
for job_id, job in data.items():
    events.append((job['arrival_time'], len(job["tasks"])))
    events.append((job["arrival_time"] + job["duration"], -len(job["tasks"])))

events.sort()

# Initialize variables for keeping track of the current number of tasks and jobs
current_tasks = 0
current_jobs = 0
task_counts = []
job_counts = []
time_points = []

# Iterate through all time points (either arrival or departure time)
for time, change in events:
    time_points.append(time)
    current_tasks += change
    task_counts.append(current_tasks)
    current_jobs += 1 if change > 0 else -1
    job_counts.append(current_jobs)

with open('../experiments/AsyncPeriodicMeasuredMTTRBenefitScheduler_short_long_rand_checkpoint/report.json', 'r') as file:
    data = json.load(file)

reconfiguration_history = data['scheduler']['global_reconfig_history']
reconfiguration_occur = [event["time"] for event in reconfiguration_history]
print(reconfiguration_occur)

# Plotting
plt.step(time_points, task_counts, where='post', label='Tasks')
plt.step(time_points, job_counts, where='post', label='Jobs')

for reconfig_time in reconfiguration_occur:
    plt.axvline(x=reconfig_time, color='r', linestyle='--', label='Reconfiguration')

plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Number of Tasks and Jobs in the System Over Time')
plt.grid(True)
plt.savefig('num_of_task_and_jobs_over_time_same_scale.png')
