import pandas as pd
import os

# Define the input and output directories
input_dir = "B:\dataset_rechts_csv_w_matched_IDs"  # Replace with your input directory path
output_dir = "B:\dataset_rechts_csv_w_matched_IDs_splitted"  # Replace with your output directory path

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through each file in the input directory
for file_name in os.listdir(input_dir):
    # Process only files with a specific extension (e.g., .txt)
    if file_name.endswith(".csv"):
        # Construct the full input and output paths
        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.csv")
        
        # Read and process the file
        with open(input_file_path, 'r') as file:
            data = [line.strip().split(',') for line in file]

        # Convert the list of lists into a DataFrame
        df = pd.DataFrame(data)

        # Save the processed data to a new CSV file
        df.to_csv(output_file_path, index=False)
        print(f"Processed and saved: {output_file_path}")
