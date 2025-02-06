import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats

# Read JSON data from file
with open('../experiments/AsyncPeriodicConstantScheduler_short_long_rand_checkpoint/report.json', 'r') as file:
    data = json.load(file)

reconfiguration_history = data['scheduler']['global_reconfig_history']
cost_before_optimization = [event["pre_optimized_global_migration_cost"] for event in reconfiguration_history]
cost_after_optimization = [event["global_migration_cost"] for event in reconfiguration_history]
improvement = [before - after for before, after in zip(cost_before_optimization, cost_after_optimization)]

# Plotting
plt.plot([event["time"] for event in reconfiguration_history], cost_before_optimization, label='Before Optimization')
plt.plot([event["time"] for event in reconfiguration_history], cost_after_optimization, label='After Optimization')
plt.xlabel('Time')
plt.ylabel('Improvement')
plt.title('Improvement in Migration Cost Over Time')
plt.savefig('migration_cost_improvement_over_time.png')

# print improvement stats
print(f"number of reconfigurations: {len(improvement)}")
print(f"Mean improvement: {np.mean(improvement)}")
print(f"Median improvement: {np.median(improvement)}")
print(f"Standard deviation of improvement: {np.std(improvement)}")
print(f"Minimum improvement: {np.min(improvement)}")
print(f"Maximum improvement: {np.max(improvement)}")
