import os
import random
import shutil


def move_random_files(source_folder, destination_folder, num_files):
    # Get a list of all files in the source folder
    all_files = os.listdir(source_folder)

    # Randomly select num_files from the list
    selected_files = random.sample(all_files, num_files)

    # Ensure the destination folder exists, create it if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Move selected files to the destination folder
    for file_name in selected_files:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.move(source_path, destination_path)
        print(f"Moved {file_name} to {destination_folder}")


# Replace these paths with your actual paths
source_folder_path = "val2017"
destination_folder_path = "train"
num_files_to_move = 2000

move_random_files(source_folder_path, destination_folder_path, num_files_to_move)
