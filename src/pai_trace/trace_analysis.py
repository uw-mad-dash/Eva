import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv("openb_pod_list_default.csv")

# use the middle 6000 data points, starting from 2000
# df = df.iloc[2000:8000]

# Convert creation_time and deletion_time to datetime objects
df['creation_time'] = pd.to_datetime(df['creation_time'], unit='s')
df['deletion_time'] = pd.to_datetime(df['deletion_time'], unit='s')

# Calculate the time difference between deletion_time and creation_time
df['time_difference'] = (df['deletion_time'] - df['creation_time']).dt.total_seconds()
# make it hours
df['time_difference'] = df['time_difference'] / 60
print(df['time_difference'].describe())
print(f"average: {df['time_difference'].mean()}")

# Plot the distribution of time difference
# plt.hist(df['time_difference'], bins=200, edgecolor='black')
# # log scale
# plt.yscale('log')
# plt.xlabel('Duration (hours)')
# plt.ylabel('Frequency (log scale)')
# plt.title('Distribution of Duration')
# plt.grid(True)
# plt.savefig('time_difference_distribution.png')
# plt.close()
# plot the cdf of time difference
sorted_data = df['time_difference'].sort_values()

# Calculate the number of jobs with duration shorter than or equal to each value
cdf_values = [(sorted_data <= value).mean() for value in sorted_data]

# Plot CDF
plt.plot(sorted_data, cdf_values)
plt.xlabel('Duration (minutes)')#
# x log scale
plt.xscale('log')
# plot a vertical line at one minute
plt.ylabel('CDF')
plt.title('CDF of Duration')
plt.grid(True)

# Save the plot
plt.savefig('time_difference_cdf.png')

# Close the plot to free up memory
plt.close()

# analyze arrival time (creation_time)
# round to integer
df = pd.read_csv("openb_pod_list_default.csv")
# round to integer
# only use middle 80% of the data
df = df.sort_values(by='creation_time')
df = df.iloc[int(len(df) * 0.1):int(len(df) * 0.9)]
df['creation_time'] = df['creation_time'].round().astype(int)
# plot histogram, with 1000 sec as bin size
print(df['creation_time'])
plt.hist(df['creation_time'], bins=range(df['creation_time'].min(), df['creation_time'].max(), 1000), edgecolor='black')
plt.xlabel('Creation Time (seconds)')
plt.ylabel('Frequency')
plt.title('Distribution of Creation Time')
plt.savefig('creation_time_distribution.png')
plt.close()

# stats for job arrival interval
print("interval:")
df['arrival_interval'] = df['creation_time'].diff()
# make hour
df['arrival_interval'] = df['arrival_interval'] / 60
print(df['arrival_interval'].describe())
print(df)
# plot 
# plot cdf
sorted_data = df['arrival_interval'].sort_values()
# Calculate the number of jobs with duration shorter than or equal to each value
cdf_values = [(sorted_data <= value).mean() for value in sorted_data]
# Plot CDF
plt.plot(sorted_data, cdf_values)
plt.xlabel('Arrival Interval (minutes)')
# x log scale
plt.xscale('log')
plt.ylabel('CDF')
plt.title('CDF of Arrival Interval')
plt.grid(True)
# Save the plot
plt.savefig('arrival_interval_cdf.png')
plt.close()

# plt.hist(df['arrival_interval'], bins=200, edgecolor='black')
# plt.yscale('log')
# plt.xlabel('Arrival Interval (minutes)')
# plt.ylabel('Frequency (log scale)')
# plt.title('Distribution of Arrival Interval')
# plt.grid(True)
# plt.savefig('arrival_interval_distribution.png')
