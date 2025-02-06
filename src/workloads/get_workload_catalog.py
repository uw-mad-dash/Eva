import os
import json

workload_dirs = [
    "a3c",
    "cyclegan",
    "gcn",
    "gpt2",
    "openfoam",
    "sage",
    "seq",
    "resnet18",
    "vit",
    "two_node_Resnet18",
    "four_node_Resnet18"
]

out_file = "workload_catalog.json"
# read config.json from these files, and compile into a single one
catalog = {}
for workload_dir in workload_dirs:
    with open(f"{workload_dir}/config.json") as file:
        workload_config = json.load(file)
        catalog[workload_dir] = workload_config

with open(out_file, 'w') as file:
    json.dump(catalog, file, indent=4)