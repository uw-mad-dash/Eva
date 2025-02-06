import json
import matplotlib.pyplot as plt

# Read the JSON file
with open("traces/physical.json", "r") as f:
    data = json.load(f)

# Extract job durations
job_durations = [data[key]["duration"] for key in data]
# convert to minutes
job_durations = [duration / 60 for duration in job_durations]

# Plot histogram of durations
plt.hist(job_durations, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Duration (minutes)')
plt.ylabel('Frequency')
plt.title('Histogram of Job Durations')
plt.grid(True)
plt.savefig("job_durations.png")
plt.show()

# print job from shortest to longest
job_durations.sort()
for duration in job_durations:
    print(duration)
print(f"Mean job duration: {sum(job_durations) / len(job_durations)}")
print(f"Max job duration: {max(job_durations)}")
print(f"Min job duration: {min(job_durations)}")

# print job name -> count
job_names = [data[key]["name"] for key in data]
job_name_counts = {}
for name in job_names:
    if name in job_name_counts:
        job_name_counts[name] += 1
    else:
        job_name_counts[name] = 1

print(job_name_counts)