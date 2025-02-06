import json
import numpy as np 
import matplotlib.pyplot as plt

def calculate_global_reconfig_ratio(report_path):
    with open(report_path, "r") as f:
        report = json.load(f)

    total_reconfig_count = len(report["scheduler"]["global_reconfig_or_not"])
    reconfig_count = report["scheduler"]["global_reconfig_or_not"].count(True)

    print(f"Reconfig count: {reconfig_count}")
    print(f"Total reconfig count: {total_reconfig_count}")
    print(f"Reconfig ratio: {reconfig_count / total_reconfig_count}")

    return reconfig_count / total_reconfig_count


calculate_global_reconfig_ratio("/home/ubuntu/eva/src/simulation_experiments/vary_migration_overhead/EVAScheduler_default_1x_migration_overhead/report.json")
calculate_global_reconfig_ratio("/home/ubuntu/eva/src/simulation_experiments/vary_migration_overhead/EVAScheduler_default_1.2x_migration_overhead/report.json")
calculate_global_reconfig_ratio("/home/ubuntu/eva/src/simulation_experiments/vary_migration_overhead/EVAScheduler_default_1.4x_migration_overhead/report.json")
calculate_global_reconfig_ratio("/home/ubuntu/eva/src/simulation_experiments/vary_migration_overhead/EVAScheduler_default_1.6x_migration_overhead/report.json")
calculate_global_reconfig_ratio("/home/ubuntu/eva/src/simulation_experiments/vary_migration_overhead/EVAScheduler_default_1.8x_migration_overhead/report.json")
calculate_global_reconfig_ratio("/home/ubuntu/eva/src/simulation_experiments/vary_migration_overhead/EVAScheduler_default_2x_migration_overhead/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_test/AsyncPeriodicBenefitScheduler_80short_20long_20min_1000jobs/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_test/AsyncPeriodicBenefitScheduler_80short_20long_20min_1000jobs_0.032_migration_overhead/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_test/AsyncPeriodicBenefitScheduler_80short_20long_20min_1000jobs_0.048_migration_overhead/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_test/AsyncPeriodicBenefitScheduler_80short_20long_20min_1000jobs_0.064_migration_overhead/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_test/AsyncPeriodicBenefitScheduler_80short_20long_20min_1000jobs_0.08_migration_overhead/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_test/AsyncPeriodicBenefitScheduler_80short_20long_20min_1000jobs_0.096_migration_overhead/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_pai/AsyncPeriodicBenefitScheduler_80short_20long_20min_500jobs_0.01_migration_overhead/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_pai/AsyncPeriodicBenefitScheduler_80short_20long_20min_500jobs_0.015_migration_overhead/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_pai/AsyncPeriodicBenefitScheduler_80short_20long_20min_500jobs_0.02_migration_overhead/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_pai/AsyncPeriodicBenefitScheduler_80short_20long_20min_500jobs_0.025_migration_overhead/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_pai/AsyncPeriodicBenefitScheduler_80short_20long_20min_500jobs_0.03_migration_overhead/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_pai/AsyncPeriodicBenefitScheduler_80short_20long_20min_500jobs_0.035_migration_overhead/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_pai/AsyncPeriodicBenefitScheduler_80short_20long_20min_500jobs_0.04_migration_overhead/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_pai/AsyncPeriodicBenefitScheduler_80short_20long_20min_500jobs_0.045_migration_overhead/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_pai/AsyncPeriodicBenefitScheduler_80short_20long_20min_500jobs_0.05_migration_overhead/report.json")
# calculate_global_reconfig_ratio("/home/ubuntu/eva/src/experiments_pai/AsyncPeriodicBenefitScheduler_80short_20long_20min_500jobs_0.1_migration_overhead/report.json")

# overhead_ratio = np.arange(0.01, 0.055, 0.005)
# data = []
# for ratio in overhead_ratio:
#     # round to third decimal place
#     ratio = round(ratio, 3)
#     data.append(calculate_global_reconfig_ratio(f"/home/ubuntu/eva/src/experiments_pai/AsyncPeriodicBenefitScheduler_80short_20long_20min_500jobs_{ratio}_migration_overhead/report.json"))

# plt.plot(overhead_ratio, data, marker='o')
# plt.xlabel("Migration overhead")
# plt.ylabel("Global reconfig proportion")
# # x axis 0.01, 0.02, 0.03, 0.04, 0.05
# plt.xticks(overhead_ratio)
# # plt.title("Global reconfig ratio vs migration overhead ratio")
# plt.savefig("global_reconfig_ratio.png")
