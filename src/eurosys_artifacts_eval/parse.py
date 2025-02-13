import sys
sys.path.append('/home/ubuntu/eva/src/parse_report')
from utils import plot_instantaneous_configuration, calculate_total_cost_per_instance


def artifacts_eval():
    path = "/home/ubuntu/eva_report.json"
    plot_instantaneous_configuration(path, "instantaneous_configuration.png")
    calculate_total_cost_per_instance(path)

artifacts_eval()