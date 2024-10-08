import os
import shutil

# Define the base directory to search and the target directory for copies
base_dir = r"B:\Opel-20241008T111613Z-001\Opel\DataExport_02062022_2_prgNr8_D41Rechts\Data\2022" 
target_dir = r"B:\dataset"  # Replace with your target directory path

# Make sure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Traverse the directory tree
for root, dirs, files in os.walk(base_dir):
    # Check if the target .pqs file exists in the current directory
    if "Seam_Seam_right__8.pqs" in files:
        # Get the measurement name from the directory path
        measurement_folder = os.path.basename(root)  # Last part of the path
        measurement_name = measurement_folder.split("_")[1] if "Measurement_" in measurement_folder else "unknown"
        
        # Define the source and destination file paths
        source_file = os.path.join(root, "Seam_Seam_right__8.pqs")
        destination_file = os.path.join(target_dir, f"{measurement_name}_Seam_Seam_right__8.pqs")
        
        # Copy the file
        shutil.copy2(source_file, destination_file)
        print(f"Copied {source_file} to {destination_file}")
