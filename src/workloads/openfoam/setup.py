import os
import shutil

def get_numerical_subdirs(directory):
    """Return a set of numerical subdirectories in the given directory."""
    try:
        subdirs = os.listdir(directory)
        # Filter and convert subdirectories to integers
        numeric_subdirs = {int(subdir) for subdir in subdirs if subdir.isdigit()}
        return numeric_subdirs
    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
        return set()
    except Exception as e:
        print(f"An error occurred while accessing {directory}: {e}")
        return set()

def find_second_largest_common_dir(base_dir, num_processors):
    """Find the second largest common numerical subdirectory among processor directories."""
    common_dirs = None
    
    for i in range(num_processors):
        processor_dir = os.path.join(base_dir, f'processor{i}')
        numeric_subdirs = get_numerical_subdirs(processor_dir)
        
        if common_dirs is None:
            common_dirs = numeric_subdirs
        else:
            common_dirs &= numeric_subdirs  # Intersection of sets
    
    if common_dirs:
        if len(common_dirs) == 1:
            return common_dirs.pop()
        else:
            # Find the second largest value in the common directories
            return sorted(common_dirs)[-2]
    else:
        return None

def remove_larger_dirs(base_dir, num_processors, start_step_dir):
    """Remove numerical directories larger than the smallest common numerical directory."""
    for i in range(num_processors):
        processor_dir = os.path.join(base_dir, f'processor{i}')
        numeric_subdirs = get_numerical_subdirs(processor_dir)
        
        for subdir in numeric_subdirs:
            if subdir > start_step_dir:
                subdir_path = os.path.join(processor_dir, str(subdir))
                print(f"Removing directory: {subdir_path}")
                shutil.rmtree(subdir_path)

# Example usage:
base_directory = '.'  # Change this to the base directory containing processor directories
num_processors = 8  # processor0 to processor7

start_step_dir = find_second_largest_common_dir(base_directory, num_processors)
if start_step_dir is not None:
    print(f'The second largest common numerical directory is: {start_step_dir}')
    remove_larger_dirs(base_directory, num_processors, start_step_dir)
else:
    print('No common numerical directories found.')
