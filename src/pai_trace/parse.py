file_path = 'openb_pod_list_default.csv'
import csv
import matplotlib.pyplot as plt
import numpy as np
import math

# Define a function to filter jobs with num_gpu=0 and plot their CPU and memory
def plot_cpu_memory_for_jobs_without_gpu(file_path):
    # Initialize lists to store CPU and memory values
    cpu_values = []
    memory_values = []

    # Open the CSV file
    with open(file_path, 'r') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(csv_file)
        
        # Iterate over each row in the CSV file
        cpu_job_num = 0
        mem_job_num = 0
        for row in csv_reader:
            # Get the task ID from the row
            task_id = row['name']
            
            # Extract the task index from the task ID
            task_index = int(task_id.split('-')[-1])
            
            # Check if task index is within the desired range (4000-5000)
            if 0 <= task_index <= 10000:
                # Get the num_gpu value from the row and convert it to an integer
                num_gpu = int(row['num_gpu'])
                
                # Check if num_gpu is equal to 0
                if num_gpu == 0:
                    # Calculate CPU value (cpu_milli / 1000)
                    cpu = int(row['cpu_milli']) / 1000
                    
                    # Calculate memory value in GB (memory_mib / 1024)
                    memory = int(row['memory_mib']) / 1024

                    if memory > 2 * cpu:
                        mem_job_num += 1
                    else:
                        cpu_job_num += 1
                    
                    # Append CPU and memory values to the respective lists
                    cpu_values.append(cpu)
                    memory_values.append(memory)
    
    print(f"Jobs with memory > 2 * CPU: {mem_job_num / 8107}")
    print(f"Jobs with memory <= 2 * CPU: {cpu_job_num / 8107}")
    # Plot CPU and memory values as a scatterplot
    plt.figure(figsize=(8, 6))
    plt.scatter(cpu_values, memory_values, marker='o', color='blue', label='Jobs without GPUs')
    
    # Plot the line y = 2x
    x_values = np.linspace(0, max(cpu_values), 100)  # Use max(cpu_values) to ensure line is plotted within the scatterplot
    y_values = 2 * x_values
    plt.plot(x_values, y_values, color='red', label='y = 2x')
    
    plt.xlabel('CPU (number of CPUs)')
    plt.ylabel('Memory (GB)')
    plt.title('CPU vs Memory for Jobs without GPUs (Tasks 4000-5000)')
    plt.grid(True)
    plt.legend()
    # plt.show()

def find_high_resource_jobs(file_path):
    # Open the CSV file
    with open(file_path, 'r') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(csv_file)

        # Counters for high resource jobs
        high_cpu_jobs = 0
        high_mem_jobs = 0

        # Iterate over each row in the CSV file
        for row in csv_reader:
            num_gpu = math.ceil(float(row['num_gpu']))
            num_cpus = math.ceil(int(row['cpu_milli']) / 1000)
            memory_gb = math.ceil(int(row['memory_mib']) / 1024)

            # Check for 8-GPU jobs with more than 64 CPUs
            if num_gpu == 8 and num_cpus > 64:
                high_cpu_jobs += 1

            # Check for 8-GPU jobs with more than 488GB of RAM
            if num_gpu == 8 and memory_gb > 488:
                high_mem_jobs += 1

    return high_cpu_jobs, high_mem_jobs

def plot_gpu_cpu_ram_for_jobs(file_path):
    # Initialize lists to store GPU, CPU, and RAM values
    # Initialize lists to store GPU, CPU, and RAM values
    gpu_values = []
    cpu_values = []
    ram_values = []

    # Open the CSV file
    with open(file_path, 'r') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(csv_file)
        
        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Get the task ID from the row
            task_id = row['name']
            
            # Extract the task index from the task ID
            task_index = int(task_id.split('-')[-1])
            
            # Check if task index is within the desired range (4000-5000)
            if True:
                # Get the GPU, CPU, and RAM values from the row and convert them to integers
                num_gpu = math.ceil(float(row['num_gpu']))
                num_cpus = math.ceil(int(row['cpu_milli']) / 1000)
                memory_gb = math.ceil(int(row['memory_mib']) / 1024)
                
                # Append GPU, CPU, and RAM values to their respective lists
                gpu_values.append(num_gpu)
                cpu_values.append(num_cpus)
                ram_values.append(memory_gb)
    
    # Plot 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(gpu_values, cpu_values, ram_values, marker='o', color='blue')

    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('CPU (GHz)')
    ax.set_zlabel('RAM (GB)')
    ax.set_title('3D Plot of GPU, CPU, and RAM for Jobs (Tasks 4000-5000)')
    
    plt.show()

# Call the function to filter jobs with num_gpu=0 and plot CPU and memory
plot_cpu_memory_for_jobs_without_gpu(file_path)
a, b = find_high_resource_jobs(file_path)
print(f"Number of 8-GPU jobs with more than 64 CPUs: {a}")
print(f"Number of 8-GPU jobs with more than 488GB of RAM: {b}")
plot_gpu_cpu_ram_for_jobs(file_path)


# Path to your CSV file

# Call the function to filter jobs with num_gpu=0 and plot CPU and memory
# plot_cpu_memory_for_jobs_without_gpu(file_path)
