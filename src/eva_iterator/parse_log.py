import re

def parse_log_file(file_path):
    throughput_values = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'\[.*?\] \[.*?\] Getting throughput - duration: \d+\.\d+, steps: \d+, throughput: (\d+\.\d+)', line)
            if match:
                throughput = float(match.group(1))
                throughput_values.append(throughput)
    return throughput_values

def calculate_stats(throughput_values):
    if not throughput_values:
        return None, None, None
    min_throughput = min(throughput_values)
    max_throughput = max(throughput_values)
    avg_throughput = sum(throughput_values) / len(throughput_values)
    return min_throughput, max_throughput, avg_throughput

file_path = "eva_iterator.log"
throughput_values = parse_log_file(file_path)
min_throughput, max_throughput, avg_throughput = calculate_stats(throughput_values)

print("Minimum throughput:", min_throughput)
print("Maximum throughput:", max_throughput)
print("Average throughput:", avg_throughput)
