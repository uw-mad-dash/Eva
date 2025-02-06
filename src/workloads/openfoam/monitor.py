import re
import os
import time

from eva_iterator import EVAIterator
EVA_SAMPLE_DURATION = 60
EVA_ITERATOR_TEST_MODE = False
EVA_ITERATOR_TEST_MODE_LOG_PERIOD = 60

def parse_control_dict(file_path):
    # Define regular expressions for startTime and endTime
    start_time_pattern = re.compile(r'startTime\s+(\d+);')
    end_time_pattern = re.compile(r'endTime\s+(\d+);')
    
    start_time = None
    end_time = None
    
    # Read the file
    with open(file_path, 'r') as file:
        content = file.read()
        
        # Search for startTime
        start_time_match = start_time_pattern.search(content)
        if start_time_match:
            start_time = int(start_time_match.group(1))
        
        # Search for endTime
        end_time_match = end_time_pattern.search(content)
        if end_time_match:
            end_time = int(end_time_match.group(1))
    
    return start_time, end_time


def find_largest_subdir(directory):
    try:
        # List all subdirectories in the given directory
        subdirs = os.listdir(directory)
        
        # Filter out non-numeric subdirectory names and convert them to integers
        numeric_subdirs = [int(subdir) for subdir in subdirs if subdir.isdigit()]
        
        # Find and return the largest value
        if numeric_subdirs:
            largest_value = max(numeric_subdirs)
            return largest_value
        else:
            return None  # Return None if there are no numeric subdirectories
    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    start_step, end_step = parse_control_dict('system/controlDict')
    cur_step = find_largest_subdir('processor0')
    print(f'Current step: {cur_step}', flush=True)
    print(f'Start step: {start_step}', flush=True)
    print(f'End step: {end_step}', flush=True)

    iterator = EVAIterator(
        range(cur_step, end_step + 1),
        sample_duration=EVA_SAMPLE_DURATION,
        test_mode=EVA_ITERATOR_TEST_MODE,
        test_mode_log_period=EVA_ITERATOR_TEST_MODE_LOG_PERIOD
    )

    for step in iterator:
        cur_step = find_largest_subdir('processor0')
        print(f"Current step: {cur_step}, iter step: {step}", flush=True)
        while cur_step <= step and cur_step != end_step:
            print(f'Waiting for step {step} to complete...', flush=True)
            time.sleep(1)
            cur_step = find_largest_subdir('processor0')
    
    print("All steps completed.", flush=True)

if __name__ == "__main__":
    main()
