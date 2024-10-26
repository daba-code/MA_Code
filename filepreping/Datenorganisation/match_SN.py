import os
import shutil
from datetime import datetime

# Define the base directories to search
base_dirs = [
    r"B:\dataset_links",
    r"B:\dataset_rechts"
]

# Create a uniquely named target directory in B: based on the current date and time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
target_dir = rf"B:\dataset_matched_IDs_{timestamp}"

# List of measurement IDs to look for
id_list = [f"{id.zfill(8)}" for id in ["145835","145845","145868","145892","145910","145937","145943","145966","145974","145978",
           "145982","145985","145987","145998","146003","146014","146022","146036","146066","146088","146126","146138","146146",
           "146158","146161","146195","146206","146215","146225","146232","146244","146247","146258"]]


# Make sure the target directory exists
os.makedirs(target_dir, exist_ok=True)
print(f"Created target directory: {target_dir}")

# Traverse each base directory
for base_dir in base_dirs:
    print(f"Searching in directory: {base_dir}")
    for root, dirs, files in os.walk(base_dir):
        for file_name in files:
            # Check if the file matches the target pattern
            if file_name.endswith("Seam_Seam_right__8.pqs"):
                # Extract the file ID from the filename (first part before "_")
                file_id = file_name.split("_")[0]
                
                # Debug: print the extracted file ID and filename
                print(f"Checking file: {file_name}, extracted ID: {file_id}")
                
                # Check if the extracted ID is in the ID list
                if file_id in id_list:
                    source_file = os.path.join(root, file_name)
                    destination_file = os.path.join(target_dir, file_name)
                    
                    try:
                        # Attempt to copy the file to the target directory
                        shutil.copy2(source_file, destination_file)
                        print(f"Copied {source_file} to {destination_file}")
                    except Exception as e:
                        print(f"Failed to copy {source_file} to {destination_file}: {e}")
                else:
                    print(f"File ID {file_id} not in ID list; skipping file {file_name}")
