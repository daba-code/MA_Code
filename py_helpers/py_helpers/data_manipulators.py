#data manipulators.py
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import time
import math

def is_row_empty(row):
    #row one-dimensional array: row == 0 "boolsch check if every element in row is 0", np.all contains sum of boolsch check true/ false for whole row
    #print(f"Checking if row is empty: {row.values}")
    return np.all(row.values == 0)

def clear_empty_profiles(df):
    #apply is_row_empty for every row in dataframe
    #is_empty referes to "pandas series": onedimensonal data structure that contains true/ false for every row of data frame
    is_empty = df.apply(is_row_empty, axis=1)
    #is_empty == False: invert series to get start of relevant data
    #is_empty[is_empty==False] select only false rows after inverting ergo: select rows until there is no row with no data 
    #.first_vaild_index(): method to get index of first non-empty row
    first_non_empty = is_empty[is_empty == False].first_valid_index()
    #same as above, only with last index
    last_non_empty = is_empty[is_empty == False].last_valid_index()
    #cut data frame down to contain rows with data in it
    #loc: access certain area of data frame within pandas
    if first_non_empty is None or last_non_empty is None:
        return df.iloc[0:0]  # Return an empty DataFrame if all rows are empty
    return df.loc[first_non_empty:last_non_empty]

def row_counter(original_df, cleaned_df):
    no_origin_profiles = original_df.shape[0]
    no_cleaned_profiles = cleaned_df.shape[0]
    removed_profiles = no_origin_profiles - no_cleaned_profiles
    print("There are", no_origin_profiles, "original profiles.")
    print("There are", no_cleaned_profiles, "profiles after cleaning,", removed_profiles, "were removed.")

def slice_columns(df, range_1, range_2, nth_column_to_remove):
    """
    Removes two specified ranges of columns and then removes every nth column from the remaining columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    range_1 (tuple): A tuple indicating the first range of columns to remove (start_1, end_1).
    range_2 (tuple): A tuple indicating the second range of columns to remove (start_2, end_2).
    nth_column_to_remove (int): The interval of columns to remove from the remaining columns.
    
    Returns:
    pd.DataFrame: The DataFrame with the specified column ranges removed and every nth column removed.
    """
    start_1, end_1 = range_1
    start_2, end_2 = range_2
    
    # Ensure ranges are within valid bounds
    total_columns = df.shape[1]
    start_1 = max(0, start_1)
    end_1 = min(total_columns, end_1)
    start_2 = max(0, start_2)
    end_2 = min(total_columns, end_2)

    # Drop the columns in the specified ranges
    columns_to_remove = df.columns[start_1:end_1].tolist() + df.columns[start_2:end_2].tolist()
    print(f"Removing specified column ranges: {columns_to_remove}")
    
    # DataFrame after removing specified column ranges
    remaining_df = df.drop(columns=columns_to_remove)
    
    # If nth_column_to_remove is greater than 0, proceed to remove every nth column
    if nth_column_to_remove > 0:
        columns_to_keep = [
            col for i, col in enumerate(remaining_df.columns)
            if (i + 1) % nth_column_to_remove != 0
        ]
        print(f"Keeping columns after applying nth column rule: {columns_to_keep}")
        remaining_df = remaining_df[columns_to_keep]
    
    return remaining_df


def slice_rows(df, keep_every_nth_row):
    """
    Keeps every nth row instead of removing.
    """
    if keep_every_nth_row <= 0:
        return df

    # Use slicing to keep only every nth row
    df_result = df.iloc[::keep_every_nth_row]
    return df_result

def segmentation(df, num_of_segments):
    
    if num_of_segments <= 0:
        return df()
    
    #count no of rows
    rows_in_df = df.shape[0]
    #define no of rows in segment, round up to next int with math.ceil
    segment_size = math.ceil(rows_in_df / num_of_segments)
    #create segments and return
    segments=[]
    for i in range(num_of_segments):
        start_row = i * segment_size
        #use min to avoid overexceeding of last segment: min returns smaller value of both
        end_row = min((i + 1) * segment_size, rows_in_df)
        #create df for segment
            #use iloc to access rows and columns based on int-based index 
        segment_df = df.iloc[start_row:end_row]
        segments.append(segment_df)
    
    return(segments)


