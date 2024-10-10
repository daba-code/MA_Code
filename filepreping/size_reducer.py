import numpy as np 
import pandas as pd
import glob
import os
from py_helpers import row_counter, clear_empty_profiles, slice_columns, slice_rows, select_folder
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_file(file, range_1, range_2, nth_column_to_remove, nth_row_to_keep, opt_file_path):
    base_filename = os.path.basename(file)
    new_filename = f"opt_{base_filename}"
    new_filepath = os.path.join(opt_file_path, new_filename)

    try: 
        chunk = pd.read_csv(file, delimiter=";")
        # Remove empty profiles (rows with all zeros)
        opt_chunk = clear_empty_profiles(chunk)
        row_counter(chunk, opt_chunk)
        
        # Apply column slicing with specified ranges and nth column rule
        opt_chunk = slice_columns(opt_chunk, range_1, range_2, nth_column_to_remove)
        
        # Apply row slicing with specified nth row rule to keep
        opt_chunk = slice_rows(opt_chunk, nth_row_to_keep)

        # Save the processed DataFrame directly to the output folder
        opt_chunk.to_csv(new_filepath, header=False, index=False, sep=";")

        return f"Saved optimized file to: {new_filepath}"
    except Exception as e:
        return f"Error processing file {file}: {e}"

def main():
    folder_path = select_folder()  # Prompt user to select folder

    if not folder_path:  # Check if the user canceled the selection
        print("No folder selected, exiting.")
        return
    
    file_path = glob.glob(os.path.join(folder_path, "*.csv"))
    # Create a single subfolder for all optimized files
    opt_file_path = os.path.join(folder_path, "optimized_files")
    if not os.path.exists(opt_file_path):
        os.makedirs(opt_file_path)

    # User input of parameters for DataFrame manipulation
    start_1 = int(input("Enter start index for the first range of columns to remove: "))
    end_1 = int(input("Enter end index for the first range of columns to remove: "))
    start_2 = int(input("Enter start index for the second range of columns to remove: "))
    end_2 = int(input("Enter end index for the second range of columns to remove: "))
    nth_column_to_remove = int(input("Enter nth column to be removed after ranges are applied: "))
    nth_row_to_keep = int(input("Enter nth row to keep: "))

    range_1 = (start_1, end_1)
    range_2 = (start_2, end_2)

    # Initialize ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Submit tasks to the executor
        futures = []
        for file in file_path:
            future = executor.submit(process_file, file, range_1, range_2, nth_column_to_remove, nth_row_to_keep, opt_file_path)
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
