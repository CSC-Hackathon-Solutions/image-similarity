import os
import shutil
from tqdm import tqdm
import sys

def move_files_with_progress(src_dir: str, dst_dir: str):
    files = os.listdir(src_dir)
    duplicates = 0

    # create the destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # iterate over all files and move them
    for file in tqdm(files, desc="Moving files", unit="file"):
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(dst_dir, file)

        if os.path.exists(dst_file):
            duplicates += 1
        else:
            shutil.move(src_file, dst_file)

    print(f'Succesfully finished.\n{duplicates} duplicates were not moved')
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <source directory> <destination directory>")
        sys.exit(1)

    # fetch the source and destination directories from command line arguments
    source_dir = sys.argv[1]
    destination_dir = sys.argv[2]

    # use the function
    move_files_with_progress(source_dir, destination_dir)

