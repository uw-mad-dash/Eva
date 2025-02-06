import os
import time

from eva_iterator import EVAIterator
EVA_SAMPLE_DURATION = 300 # about 60 sec per block
EVA_ITERATOR_TEST_MODE = False
EVA_ITERATOR_TEST_MODE_LOG_PERIOD = 60

TODO_FILE = "ptmp/align_todo_000000"
DONE_FILE = "ptmp/align_done_000000"

def find_largest_val_in_col(file_path, col_num):
    with open(file_path, 'r') as file:
        values = [int(line.strip().split()[col_num]) for line in file]
    return max(values)

def find_cur_block_num():
    # if nothing has been processed, return 0
    if os.path.getsize(DONE_FILE) == 0:
        return 0
    return find_largest_val_in_col(DONE_FILE, 0) + 1

def find_total_block_num():
    return find_largest_val_in_col(TODO_FILE, 0) + 1

def main():
    while any(not os.path.exists(file) for file in [TODO_FILE, DONE_FILE]):
        print(f"Waiting for files to be created", flush=True)
        time.sleep(1)
    
    while any(os.path.getsize(file) == 0 for file in [TODO_FILE]):
        print(f"Waiting for files to be written", flush=True)
        time.sleep(1)

    start_block_num = find_cur_block_num()
    total_block_num = find_total_block_num()

    print(f"Start block number: {start_block_num}", flush=True)
    print(f"Total block number: {total_block_num}", flush=True)

    iterator = EVAIterator(
        range(start_block_num, total_block_num),
        sample_duration=EVA_SAMPLE_DURATION,
        test_mode=EVA_ITERATOR_TEST_MODE,
        test_mode_log_period=EVA_ITERATOR_TEST_MODE_LOG_PERIOD
    )

    for block_num in iterator:
        cur_block_num = find_cur_block_num()
        print(f"Current block number: {cur_block_num}, iter block number: {block_num}", flush=True)
        time_waited = 0
        while cur_block_num <= block_num:
            if time_waited % 10 == 0:
                print(f"Waiting for block {block_num} to complete...", flush=True)
            time.sleep(1)
            time_waited += 1
            cur_block_num = find_cur_block_num()

    print("All blocks have been processed.", flush=True)

if __name__ == "__main__":
    main()
