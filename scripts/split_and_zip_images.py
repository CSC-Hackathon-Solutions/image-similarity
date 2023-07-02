import os
import shutil
import zipfile
from tqdm import tqdm
import glob
import math
import argparse

def split_folder(folder_path, output_folder, parts):
    # Get all files in the folder
    files = os.listdir(folder_path)
    files.sort()
    total_files = len(files)

    # Calculate how many files for each part
    files_per_part = math.ceil(total_files / parts)

    for i in tqdm(range(parts), desc="📦 Zipping", unit="part"):
        part_folder = os.path.join(output_folder, f'part_{i+1}')
        os.makedirs(part_folder, exist_ok=True)
        
        # Select files for this part
        start_index = i * files_per_part
        end_index = start_index + files_per_part
        part_files = files[start_index:end_index]
        
        # Copy files to the part folder
        for file in part_files:
            shutil.copy(os.path.join(folder_path, file), part_folder)
            
        # Create a zip file for this part
        with zipfile.ZipFile(f'{part_folder}.zip', 'w') as zipf:
            for file in glob.glob(f'{part_folder}/*'):
                zipf.write(file)
                
        # Remove the part folder
        shutil.rmtree(part_folder)

    print("🎉 Zipping process completed!")


parser = argparse.ArgumentParser(description='🖥 Process command line arguments.')

# Add command line arguments
parser.add_argument('-i', '--input_folder', type=str, required=True, help='📤 Path to your input folder')
parser.add_argument('-o', '--output_folder', type=str, required=True, help='📥 Path to your output folder')
parser.add_argument('-n', '--num_parts', type=int, required=True, help='🤔 Number of parts to split the folder into')

# Parse arguments
args = parser.parse_args()

# Call split_folder function with parsed arguments
split_folder(args.input_folder, args.output_folder, args.num_parts)

