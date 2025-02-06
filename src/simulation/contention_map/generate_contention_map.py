import csv
import json
from itertools import combinations_with_replacement, chain
from collections import defaultdict
from functools import reduce
import pickle

def get_pairwise_contention_info(matrix_filename, default_throughput=0.95):
    """
    Reads a CSV file and converts it to the desired JSON format.

    The CSV file is expected to look like this:

    , model1, model2, model3
    model1, 0.2, 0.5, x
    model2, 0.4, 0.3, 0.7
    model3, x, 0.9, 0.8
    """
    with open(matrix_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        throughput_dict = {}
        
        for row in reader:
            target_workload = row[0]
            for i, value in enumerate(row[1:], start=1):
                if value == 'x':
                    value = default_throughput
                colocated_workload = header[i]
                throughput = float(value)
                throughput_dict.setdefault(target_workload, {})[colocated_workload] = throughput
        
        tmp_dict = {}
        for target_workload, colocated_workloads in throughput_dict.items():
            for colocated_workload, throughput in colocated_workloads.items():
                tmp_dict[(target_workload, colocated_workload)] = throughput
        
        merged_dict = {}
        for (workload1, workload2), throughput in tmp_dict.items():
            if workload1 == workload2:
                merged_dict[(workload1, workload2)] = [throughput, throughput]
                continue
            if (workload2, workload1) in merged_dict:
                merged_dict[(workload2, workload1)].append(throughput)
            else:
                merged_dict[(workload1, workload2)] = [throughput]
        
        json_dict = {}
        for i, (workloads, throughputs) in enumerate(merged_dict.items()):
            json_dict[i] = {
                'workloads': workloads,
                'throughputs': throughputs
            }
        
        return json_dict

def get_throughput(w1, w2, throughput_data):
    for value in throughput_data.values():
        workloads = value["workloads"]
        throughputs = value["throughputs"]
        if set([w1, w2]) == set(workloads):
            return throughputs[workloads.index(w1)]
    return None

def generate_extended_contention_info(data, max_n=10):
    workloads = list(set(chain.from_iterable([d["workloads"] for d in data.values()])))
    throughput_data = defaultdict(dict)

    for workload in workloads:
        throughput_data[1][(workload,)] = [1.0]
    
    for d in data.values():
        combo = d["workloads"]
        throughputs = d["throughputs"]
        throughput_data[2][tuple(combo)] = throughputs
    
    for n in range(3, max_n + 1):
        # for n>= 8, only deal with cpu jobs
        if n > 10:
            workloads = ["gcn_node0", "a3c_node0", "openfoam_node0", "seq_node0"]
        for combo in combinations_with_replacement(workloads, n):
            combo_throughput = {}
            for i, w in enumerate(combo):
                other_workloads = list(combo[:i] + combo[i+1:])
                throughput_product = reduce(lambda x, y: x * y, [get_throughput(w, other, data) for other in other_workloads], 1)
                combo_throughput[w] = throughput_product
            throughput_data[n][combo] = [combo_throughput[w] for w in combo]
        print(f"Finished n={n}, number of entries: {len(throughput_data[n])}")
    
    extended_data = {}
    id = 0
    for n, combo_data in throughput_data.items():
        for workloads, throughputs in combo_data.items():
            extended_data[id] = {
                "workloads": workloads,
                "throughputs": throughputs
            }
            id += 1
    
    return extended_data

def create_contention_map(contention_info):
    """
    {(w1, w2, w3): [t1, t2, t3]}
    """
    contention_map = {}
    for i, value in contention_info.items():
        workloads = tuple(sorted(value["workloads"]))
        throughputs = value["throughputs"]
        contention_map[workloads] = throughputs
    
    return contention_map

# def create_contention_map(contention_info):
#     """
#     {
#         "resnet": {
#             ("a3c", "sage", "sage"): 0.95,
#             ...
#         },
#         ...
#     }
#     """
#     contention_map = {}
#     for i, value in contention_info.items():
#         workloads = value["workloads"]
#         throughputs = value["throughputs"]
#         for j, workload in enumerate(workloads):
#             if workload not in contention_map:
#                 contention_map[workload] = {}
#             other_workloads = workloads[:j] + workloads[j+1:]
#             # sort other_workloads to ensure consistent key order
#             other_workloads = tuple(sorted(other_workloads))
#             contention_map[workload][other_workloads] = throughputs[j]
    
#     return contention_map



def main():
    matrix_filename = 'contention_matrix.csv'  # Replace with your actual CSV file path
    extended_json_filename = 'full_contention_info.json'
    contention_map_filename = 'contention_map.pkl'
    
    # Step 1: Convert CSV to JSON
    print("Generating pairwise contention info...")
    pairwise_contention_data = get_pairwise_contention_info(matrix_filename, default_throughput=0.95)
    
    # Step 3: Generate extended throughput data
    print("Generating extended contention info...")
    extended_throughput_data = generate_extended_contention_info(pairwise_contention_data, max_n=25)
    
    # # Step 4: Save the extended data to a JSON file
    # with open(extended_json_filename, "w") as f:
    #     f.write(json.dumps(extended_throughput_data, indent=4))
    
    print(f"Number of entries: {len(extended_throughput_data)}")

    # # Step 5: Create a contention map
    print("Creating contention map...")
    contention_map = create_contention_map(extended_throughput_data)
    # save to pickle
    with open(contention_map_filename, "wb") as f:
        pickle.dump(contention_map, f)

if __name__ == '__main__':
    main()
