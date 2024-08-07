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

def slice_columns(df, percentage_of_columns_to_remove, nth_column_to_remove):
    #count columns
    total_columns = df.shape[1]
    print(f"Total columns: {total_columns}")
    #no of total columns to be removed in beginning and end of dataframe
    if percentage_of_columns_to_remove > 0:
        columns_to_remove = int(total_columns * (percentage_of_columns_to_remove / 100))
        #print(f"Columns to remove from start and end: {columns_to_remove}")
        #get no of columns to be removed in beginning and end of df
        half_columns_to_remove = columns_to_remove // 2

        if columns_to_remove > 0: 
            #remove from start
            start_of_columns_to_keep = df.columns[half_columns_to_remove:total_columns]
            #remove from end
            end_of_columns_to_keep = start_of_columns_to_keep[:-half_columns_to_remove]
            #print(f"Columns after removing from start and end: {end_of_columns_to_keep}")
        else:
            end_of_columns_to_keep = df.columns
        #return df of original df only containing the selected columns
        remaining_columns = df[end_of_columns_to_keep]
        #print(f"Remaining columns: {remaining_columns.columns}")
    else:
        remaining_columns = df

    if nth_column_to_remove <=0:
        return remaining_columns
    #remove every nth column of remaining dataframe
    
    #mask for storing boolsch values to indicate whether column should be kept or removed 
    columns_to_keep = []
    #iterate over the columns of the remaining DataFrame with their index using enumerate
    for i, column in enumerate(remaining_columns.columns):
        #create 1-based index
        column_index = i + 1
        #check if column_index is not a multiple of nth column
        if column_index % nth_column_to_remove != 0:
            #if condition true, then append column name to the columns_to_keep list
            columns_to_keep.append(column)
        else:
            print(f"Removing column: {column} (index: {i})")
    
    #apply mask to dataframe
    #.loc syntax: df.loc[row_indexer, column_indexer]; ":" in the row position refers to select all rows
    df_result = remaining_columns[columns_to_keep]
    #print(f"Final columns: {df_result.columns}")
    return(df_result)

def slice_rows(df, nth_row_to_remove):

    if nth_row_to_remove <= 0:
        return(df)
    
    #count rows
    total_remaining_rows = df.shape[0]
    #mask for storing boolsch values to indicate whether row should be kept or removed
    rows_to_keep = []
    #iterate over rows with their index using enum
    for i in range(total_remaining_rows):
        #create 1-based index
        row_index = i + 1
        #check if row_index is not a multiple of nth row
        if row_index % nth_row_to_remove != 0:
            rows_to_keep.append(True)
        else:
            rows_to_keep.append(False)
    #apply mask to df
    df_result = df[rows_to_keep]
    return(df_result)

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


