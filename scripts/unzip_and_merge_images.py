import os
import shutil
import zipfile
from tqdm import tqdm
import glob
import argparse

def merge_folders(zipped_parts_folder, output_folder):
    # Get all zip files in the folder
    zip_files = [f for f in os.listdir(zipped_parts_folder) if f.endswith('.zip')]
    zip_files.sort()

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for i in tqdm(range(len(zip_files)), desc="📦 Unzipping", unit="part"):
        part_file = os.path.join(zipped_parts_folder, zip_files[i])

        # Extract zip file to a temporary folder
        with zipfile.ZipFile(part_file, 'r') as zipf:
            temp_dir = os.path.join(output_folder, f'temp_{i}')
            os.makedirs(temp_dir, exist_ok=True)
            zipf.extractall(temp_dir)

            # Go through each sub-directory in the temp_dir
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    old_file_path = os.path.join(root, file)
                    # New file path in the output_folder
                    new_file_path = os.path.join(output_folder, file)

                    # Move each file to the output_folder
                    shutil.move(old_file_path, new_file_path)

            # Remove temporary directory
            shutil.rmtree(temp_dir)

    print("🎉 Unzipping and merging process completed!")



# create parser
parser = argparse.ArgumentParser(description='Process command line arguments.')

# Add command line arguments
parser.add_argument('-i', '--input_folder', type=str, required=True, help='📤 Path to your input folder')
parser.add_argument('-o', '--output_folder', type=str, required=True, help='📥 Path to your output folder')

# Parse arguments
args = parser.parse_args()

# Call merge_folders function with parsed arguments
merge_folders(args.input_folder, args.output_folder)
