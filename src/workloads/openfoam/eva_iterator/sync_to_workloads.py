import os
import shutil

def copy_folder_to_directories(src_folder, dst_directories):
    """
    Copies a folder to multiple destination directories.

    :param src_folder: The source folder to be copied.
    :param dst_directories: A list of destination directories.
    """
    if not os.path.exists(src_folder):
        raise ValueError(f"Source folder '{src_folder}' does not exist.")

    for dst_dir in dst_directories:
        # Create the full path for the destination folder
        dst_folder = os.path.join(dst_dir, os.path.basename(src_folder))
        try:
            # Copy the source folder to the destination folder
            shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
            print(f"Copied '{src_folder}' to '{dst_folder}'")
        except Exception as e:
            print(f"Failed to copy '{src_folder}' to '{dst_folder}': {e}")

if __name__ == "__main__":
    # Example usage
    source_folder = "/home/ubuntu/eva/src/eva_iterator"
    destination_directories = [
        "/home/ubuntu/gcn",
        "/home/ubuntu/openfoam",
        "/home/ubuntu/mount/template_prep/a3c",
        "/home/ubuntu/mount/template_prep/cyclegan",
        "/home/ubuntu/mount/template_prep/gcn",
        "/home/ubuntu/mount/template_prep/gpt2/src",
        "/home/ubuntu/mount/template_prep/sage",
        "/home/ubuntu/mount/template_prep/single",
        "/home/ubuntu/mount/template_prep/vit",
    ]

    copy_folder_to_directories(source_folder, destination_directories)
