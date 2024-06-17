import numpy as np 
import pandas as pd
import glob
import os
from py_helpers import row_counter, clear_empty_profiles, slice_columns, slice_rows
from tqdm import tqdm

# data to be handled
folder_path = r"C:\Users\dabac\Desktop\testfiles_for_data"
file_path = glob.glob(os.path.join(folder_path, "*.csv"))

# create subfolder for opt files
opt_file_path = os.path.join(folder_path, "optimized_files")
if not os.path.exists(opt_file_path):
    os.makedirs(opt_file_path)

chunksize = 10

def process_file(file, chunksize, data_chunks):
    base_filename = os.path.basename(file)
    new_filename = f"opt_{base_filename}"
    new_filepath = os.path.join(opt_file_path, new_filename)

    try: 
        if file not in data_chunks:
            data_chunks[file] = []

        for chunk in pd.read_csv(file, delimiter=";", chunksize=chunksize):
            # Remove empty profiles (rows with all zeros)
            opt_chunk = clear_empty_profiles(chunk)
            row_counter(chunk, opt_chunk)
            data_chunks[file].append(opt_chunk)

        combined_df = pd.concat(data_chunks[file], ignore_index=True)
        combined_df.to_csv(new_filepath, header=False, index=False, sep=";")
        print(f"Saved optimized file to: {new_filepath}")
        os.startfile(new_filepath)
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        print(f"Error writing file {new_filepath}: {e}")

def main(folder_path, file_path, chunksize):
    data_chunks = {}
    try:
        with tqdm(total=len(file_path), desc="processing file") as pbar:
            for file in file_path:
                process_file(file, chunksize, data_chunks)
                pbar.update(1)
    except KeyboardInterrupt:
        print("process ended by user")

#only execute within this file
if __name__ == "__main__":
    main(folder_path, file_path, chunksize)