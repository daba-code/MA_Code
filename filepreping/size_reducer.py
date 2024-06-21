import numpy as np 
import pandas as pd
import glob
import os
from py_helpers import row_counter, clear_empty_profiles, slice_columns, slice_rows, select_folder, segmentation
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_file(file, percentage_of_columns_to_remove, nth_column_to_remove, nth_row_to_remove, opt_file_path, num_segments):
    base_filename = os.path.basename(file)
    file_stem = os.path.splitext(base_filename)[0]
    subfolder_path = os.path.join(opt_file_path, file_stem)

    # Erstellen des Unterordners, falls er nicht existiert
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    new_filename = f"opt_{base_filename}"
    new_filepath = os.path.join(subfolder_path, new_filename)

    try: 
        chunk = pd.read_csv(file, delimiter=";")
        # Remove empty profiles (rows with all zeros)
        opt_chunk = clear_empty_profiles(chunk)
        row_counter(chunk, opt_chunk)
        opt_chunk = slice_columns(opt_chunk, percentage_of_columns_to_remove, nth_column_to_remove)
        opt_chunk = slice_rows(opt_chunk, nth_row_to_remove)

        #apply segmentation
        segments = segmentation(opt_chunk, num_segments)
        for idx, segment in enumerate(segments):
            segment_filename = f"{new_filename.split('.csv')[0]}_segment_{idx + 1}.csv"
            segment_filepath = os.path.join(opt_file_path, segment_filename)
            segment.to_csv(segment_filepath, header=False, index=False, sep=";")

        return f"Saved optimized file to: {subfolder_path}"
    except Exception as e:
        return f"Error processing file {file}: {e}"

def main():
    folder_path = select_folder()  # Prompt user to select folder

    if not folder_path:  # Check if the user canceled the selection
        print("No folder selected, exiting.")
        return
    
    file_path = glob.glob(os.path.join(folder_path, "*.csv"))
    # create subfolder for opt files
    opt_file_path = os.path.join(folder_path, "optimized_segmented_files")
    if not os.path.exists(opt_file_path):
        os.makedirs(opt_file_path)

    # User input of parameters for dataframe manipulation
    percentage_of_columns_to_remove = int(input("Enter percentage of columns to be removed: "))
    nth_column_to_remove = int(input("Enter nth column to be removed: "))
    nth_row_to_remove = int(input("Enter nth row to be removed: "))
    num_segments = int(input("Enter number of segments to split to:"))

    # Initialize ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Submit tasks to the executor
        futures = []
        for file in file_path:
            future = executor.submit(process_file, file, percentage_of_columns_to_remove, nth_column_to_remove, nth_row_to_remove, opt_file_path, num_segments)
            futures.append(future)
        
        # Process the results as they complete
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                result = future.result()
                print(result)
        except KeyboardInterrupt:
            print("Process ended by user")

# Only execute within this file
if __name__ == "__main__":
    main()
