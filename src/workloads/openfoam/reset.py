import os
import shutil

def remove_numerical_dirs_except_zero(directory):
    try:
        # List all entries in the given directory
        entries = os.listdir(directory)
        
        for entry in entries:
            entry_path = os.path.join(directory, entry)
            # Check if the entry is a directory and its name is numeric but not "0"
            if os.path.isdir(entry_path) and entry.isdigit() and entry != "0":
                print(f"Removing directory: {entry_path}")
                # Remove the directory and all its contents
                shutil.rmtree(entry_path)
                
    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
for i in range(8):
    remove_numerical_dirs_except_zero(f"processor{i}")
