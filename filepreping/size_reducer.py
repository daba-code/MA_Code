import numpy as np 
import pandas as pd
import glob
import os
from py_helpers import row_counter, clear_empty_profiles, slice_columns, slice_rows
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog

def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory()  # Open the dialog to choose a directory
    root.destroy()  # Ensure the root tkinter instance is destroyed after selection
    return folder_selected

def process_file(file, chunksize, data_chunks, percentage_of_columns_to_remove, nth_column_to_remove, nth_row_to_remove, opt_file_path):
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
            opt_chunk = slice_columns(opt_chunk, percentage_of_columns_to_remove, nth_column_to_remove)
            opt_chunk = slice_rows(opt_chunk, nth_row_to_remove)
            data_chunks[file].append(opt_chunk)

        combined_df = pd.concat(data_chunks[file], ignore_index=True)
        combined_df.to_csv(new_filepath, header=False, index=False, sep=";")
        print(f"Saved optimized file to: {new_filepath}")
        #os.startfile(new_filepath)
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        
def main():
    #data to be modified 
    folder_path = select_folder()  # Prompt user to select folder

    if not folder_path:  # Check if the user canceled the selection
        print("No folder selected, exiting.")
        return
    
    file_path = glob.glob(os.path.join(folder_path, "*.csv"))
    # create subfolder for opt files
    opt_file_path = os.path.join(folder_path, "optimized_files")
    if not os.path.exists(opt_file_path):
        os.makedirs(opt_file_path)

    #user input of parameters for dataframe manipulation
    percentage_of_columns_to_remove = int(input("Enter percentage of columns to be removed:"))
    nth_column_to_remove = int(input("Enter nth column to be removed:"))
    nth_row_to_remove = int(input("Enter nth row to be removed:"))

    chunksize = 10

    data_chunks = {}
    try:
        with tqdm(total=len(file_path), desc="processing file") as pbar:
            for file in file_path:
                process_file(file, chunksize, data_chunks, percentage_of_columns_to_remove, nth_column_to_remove, nth_row_to_remove, opt_file_path)
                pbar.update(1)
    except KeyboardInterrupt:
        print("process ended by user")

#only execute within this file
if __name__ == "__main__":
    main()