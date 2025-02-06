#!/usr/bin/env python3
import re
import sys
import os

# Small struct to keep data about each logical processor
class Processor:
    def __init__(self, processor, core, node, socket):
        self.processor = processor
        self.core = core
        self.node = node
        self.socket = socket

    def __lt__(self, other):
        return self.processor < other.processor

def shift(l, n):
    return l[n:] + l[:n]

def sort_and_shift(l):
    return shift(sorted(l), 1)

# Given a processor ID, return its corresponding NUMA node
def determine_node(processor):
    files = os.listdir(f"/sys/devices/system/cpu/cpu{processor}")
    for f in files:
        match = re.search(r"node(\d+)", f)
        if match:
            return match.group(1)

    # No NUMA node found - return -1
    return "-1"

# Parse /proc/cpuinfo
def parse_processor_info(raw):
    info = {}

    for raw_processor in raw.split("\n\n")[:-1]:
        match = re.search(r"processor\s+:\s(\d+)", raw_processor)
        processor = int(match.group(1))

        match = re.search(r"physical id\s+:\s(\d+)", raw_processor)
        socket = int(match.group(1))

        match = re.search(r"core id\s+:\s(\d+)", raw_processor)
        core = int(match.group(1))

        # Determine memory node
        node = int(determine_node(processor))

        processor_obj = Processor(processor, core, node, socket)

        if socket not in info:
            info[socket] = {}

        if node not in info[socket]:
            info[socket][node] = {}

        if core not in info[socket][node]:
            info[socket][node][core] = []

        info[socket][node][core].append(processor_obj)

    return info

def print_topology(info):
    sockets = 0
    cores = 0
    processors = 0
    nodes = 0

    for socket in sorted(info.keys()):
        sockets += 1
        print(f"Package {socket}")

        for node in sorted(info[socket].keys()):
            nodes += 1
            print(f"\tNUMA node {node}")

            for core in sorted(info[socket][node].keys()):
                cores += 1
                print(f"\t\tPhysical core {core}")

                for processor in sorted(info[socket][node][core]):
                    processors += 1
                    print(f"\t\t\tProcessor: {processor.processor}")

    print(f"\nIn total, there are {sockets} physical packages (sockets), {nodes} NUMA nodes, {cores} physical cores and {processors} hardware threads\n")

# Avoid using processor 0, only put at the very last,
# since it's normally used by the OS, interrupts etc
def scatter(socket, node, core, processor, info):
    if socket >= len(info):
        return scatter(0, node + 1, core, processor, info)

    socket_id = sort_and_shift(list(info.keys()))[socket]
    if node >= len(info[socket_id]):
        return scatter(socket, 0, core + 1, processor, info)

    node_id = sort_and_shift(list(info[socket_id].keys()))[node]
    if core >= len(info[socket_id][node_id]):
        return scatter(socket, node, 0, processor + 1, info)

    core_id = sort_and_shift(list(info[socket_id][node_id].keys()))[core]
    if processor >= len(info[socket_id][node_id][core_id]):
        return []

    target = sort_and_shift(info[socket_id][node_id][core_id])[processor].processor
    return [target] + scatter(socket + 1, node, core, processor, info)

def compact(info):
    mapping = []
    # For each socket
    for socket in sort_and_shift(list(info.keys())):
        # For each node
        for node in sort_and_shift(list(info[socket].keys())):
            # For each core
            for core in sort_and_shift(list(info[socket][node].keys())):
                # For each hardware thread
                for thread in sort_and_shift(info[socket][node][core]):
                    mapping.append(thread.processor)

    return mapping

def get_cpu_preference_list():
    with open("/proc/cpuinfo") as f:
        raw_info = f.read()

    info = parse_processor_info(raw_info)
    return scatter(0, 0, 0, 0, info)

def get_num_physical_cores():
    with open("/proc/cpuinfo") as f:
        raw_info = f.read()

    info = parse_processor_info(raw_info)
    count = 0
    for socket in info:
        for node in info[socket]:
            for core in info[socket][node]:
                count += 1
    return count

def main():
    if len(sys.argv) > 2:
        print("Error: Too many arguments")
        print(f"Usage: {sys.argv[0]} [affinity mapping]")
        sys.exit(1)

    with open("/proc/cpuinfo") as f:
        raw_info = f.read()

    info = parse_processor_info(raw_info)

    if len(sys.argv) == 1:
        print_topology(info)

    # The script can also output some affinity mappings
    # For example, running ./cpu-topology.py scatter will
    # output a scatter affinity which will distribute 
    # the cores/processes as evenly as possible
    if len(sys.argv) == 2:
        affinity = sys.argv[1]
        if affinity == "scatter":
            print(",".join(str(x) for x in scatter(0, 0, 0, 0, info)))
        elif affinity == "compact":
            print(",".join(str(x) for x in compact(info)))
        else:
            print(f"Error: Unrecognized affinity mapping '{affinity}'")

if __name__ == '__main__':
    main()
