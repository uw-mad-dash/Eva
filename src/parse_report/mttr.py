import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats

# Read JSON data from file
with open('../experiments/AsyncPeriodicConstantScheduler_short_long_rand_checkpoint/report.json', 'r') as file:
    data = json.load(file)

# Access the scheduler's global reconfiguration history
global_reconfig_history = data['scheduler']['global_reconfig_history']
actual_reconfig = [event["time"] for event in global_reconfig_history]
actual_interval = np.diff(actual_reconfig)
print(len(actual_interval))
estimated_interval = [event["mean_time_to_next_reconfig"] for event in global_reconfig_history[:-1]]
print(len(estimated_interval))
for i in range(len(estimated_interval)):
    print(f"Time: {global_reconfig_history[i]['time']}, Actual: {actual_interval[i]}, Estimated: {estimated_interval[i]}, migration_worthwhile_time: {global_reconfig_history[i]['migration_worthwhile_time']}")

# Calculate statistics
mse = mean_squared_error(actual_interval, estimated_interval)
mae = mean_absolute_error(actual_interval, estimated_interval)
r_value, p_value = stats.pearsonr(actual_interval, estimated_interval)
r_squared = r_value ** 2
rmse = np.sqrt(mse)

# Report statistics
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"Pearson's r: {r_value}, pvalue: {p_value}")
print(f"R-squared: {r_squared}")
print(f"RMSE: {rmse}")


# see how related the actual and estimated intervals are
plt.scatter(actual_interval, estimated_interval)
plt.xlabel('Actual interval')
plt.ylabel('Estimated interval')
plt.title('Actual vs Estimated Reconfiguration Interval')
# x y axis same scale, same range
# start from 0
lim = max(max(actual_interval), max(estimated_interval))
plt.xlim(0, lim)
plt.ylim(0, lim)
plt.grid(True)

# save 
plt.savefig('actual_vs_estimated_reconfig_interval.png')
