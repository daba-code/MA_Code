import numpy as np 
import pandas as pd
import glob
import os
from py_helpers import row_counter, clear_empty_profiles, slice_columns, slice_rows
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ProcessPoolExecutor, as_completed

def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory()  # Open the dialog to choose a directory
    root.destroy()  # Ensure the root tkinter instance is destroyed after selection
    return folder_selected

def process_file(file, percentage_of_columns_to_remove, nth_column_to_remove, nth_row_to_remove, opt_file_path):
    base_filename = os.path.basename(file)
    new_filename = f"opt_{base_filename}"
    new_filepath = os.path.join(opt_file_path, new_filename)

    try: 
        chunk = pd.read_csv(file, delimiter=";", dtype="int16")
        # Remove empty profiles (rows with all zeros)
        opt_chunk = clear_empty_profiles(chunk)
        row_counter(chunk, opt_chunk)
        opt_chunk = slice_columns(opt_chunk, percentage_of_columns_to_remove, nth_column_to_remove)
        opt_chunk = slice_rows(opt_chunk, nth_row_to_remove)

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
    # create subfolder for opt files
    opt_file_path = os.path.join(folder_path, "optimized_files")
    if not os.path.exists(opt_file_path):
        os.makedirs(opt_file_path)

    # User input of parameters for dataframe manipulation
    percentage_of_columns_to_remove = int(input("Enter percentage of columns to be removed: "))
    nth_column_to_remove = int(input("Enter nth column to be removed: "))
    nth_row_to_remove = int(input("Enter nth row to be removed: "))

    # Initialize ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Submit tasks to the executor
        futures = []
        for file in file_path:
            future = executor.submit(process_file, file, percentage_of_columns_to_remove, nth_column_to_remove, nth_row_to_remove, opt_file_path)
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
