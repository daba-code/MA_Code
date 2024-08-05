# import libs
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import time
import glob
import os
from py_helpers import row_counter, clear_empty_profiles

# data to be plotted
folder_path = r"C:\Users\dabac\Desktop\MA\Seam_left1_1\OK\optimized_files\Gauss_Training_Data\optimized_segmented_files\optimized_segmented_files\optimized_segmented_files\optimized_segmented_files"
file_path = glob.glob(os.path.join(folder_path, "*.csv"))
chunksize = 10000

try:
    #create dicitionary to store data chunks for each file 
        #general dictionary syntax: {key: value for item in iterable} | create dictionary by iterating over list or range and applying expressions such as functions, variables, lists
        #file = key in the dictionary, file is each element in file_path
        #[] associated value for each key, in this case every key gets an empty list
        #for file in file_paths: loop to iterate over each input file
    data_chunks = {file: [] for file in file_path}

    #optimize chunks for each file in file path
    for file in file_path:
            try:
                print(f"reading file: {file}")
                #process file in chunks
                for chunk in pd.read_csv(file, delimiter=";", chunksize=chunksize):
                    #remove empty profiles
                    opt_chunk = clear_empty_profiles(chunk)
                    row_counter(chunk, opt_chunk)
                    #append cleaned chunk to dictionary
                    data_chunks[file].append(opt_chunk)
            except Exception as error:
                print(f"error occured during processing file {file}: {error}")

    #initialize list to store total number of profiles for each file AFTER optimization
    total_profiles_per_file = []

    #iterate over each file to calculate its total profiles
    for file in file_path:
         #get number of profiles in each chunk 
            #create list that contains number of profiles in each optimized chunk "for each optimized chunk in the list get the number of profiles in that chunk and add to list" 
         profiles_in_chunk = [opt_chunk.shape[0] for opt_chunk in data_chunks[file]]
         #sum of all profiles in current file
         total_profiles = sum(profiles_in_chunk)
         #add total profiles in list of total_profiles_per_file
         total_profiles_per_file.append(total_profiles)

    #identify smallest number of profiles across all files
    min_profiles = min(total_profiles_per_file)
    #and biggest number of profiles accross all files
    max_profiles = max(total_profiles_per_file)
    print(f"total profiles per file after removing empty profiles: {total_profiles_per_file}")
    print(f"smallest number of profiles across all files (optimized): {min_profiles}")
    print(f"biggest number of profiles across all files (optimized): {max_profiles}")

    profile_index = 0

    while profile_index < max_profiles: #loop until all profiles of every file is reached
        plt.clf() #clear figure
        plt.ylim(0, 1000) #set y-limits
        all_files_displayed = True

        #iterate over each file in file_path list
        #file_index is the index of the current file in the file path
        #file is the path of the current file being processed
        for file_index, file in enumerate(file_path): #iterate over files
            if "NOK" in file:
                color = "red"
            else:
                color = "black"
            #plot based on OK/NOK
            #print(f"File: {file}, Color: {color}")

            chunk_count = 0 #indexing chunks
            
            #iterate/ loop over each optimized chunk for the current file 
                #(data_chunk[file]) list of data chunks corresponding to the current file
            for opt_chunk in data_chunks[file]: #iterate over chunks

                chunks_list = data_chunks[file] #get list of data chhunks for current file
                end_index = chunk_count + 1 #position up to needed chunks
                chunks_up_to_current = chunks_list[:end_index] #hold all chunk from beginning up to current chunk

                #calculate total number of profiles in current and previous chunks
                total_profiles_up_to_current_chunk = sum(
                     opt_chunk.shape[0] for opt_chunk in data_chunks[file][:chunk_count+1]
                     )

                #check if current profile_index is within range of the current chunk
                if profile_index < total_profiles_up_to_current_chunk:
                    #calculate total number of profiles before current chunk
                    total_profiles_before_current_chunk = sum(
                         opt_chunk.shape[0] for opt_chunk in data_chunks[file][:chunk_count]
                    )
                    #determine local profile index within current chunk
                    local_profile_index = profile_index - total_profiles_before_current_chunk
    
                    #extract data values of current profile in row "i": opt_chunk.values[profile_index] returns value depending on row i as NumPy array
                    data_values = opt_chunk.values[local_profile_index] #DF for matplotlib, onedimensional array, contains all values of profile
                    x_values = range(len(data_values)) #length of array data_values refers to overall no. of datapoints = values on x-axis of data-values

                    # Plot the current profile
                    plt.plot(
                        x_values, data_values, color, label=f'Profile {profile_index+1} from {file}'
                        )
                    
                    #indicate that profiles are still being display
                    all_files_displayed = False

                    #exit inner loop after plotting profile
                    break

                #increment chunk counter
                chunk_count = chunk_count + 1 

        #create title with all files in file path
        file_string = ",".join(file_path)
        plt.title(f'Profile {profile_index+1} from {file_string}')
        plt.xlabel('Index der Messwerte')
        plt.ylabel('Höhenwerte')
        plt.legend
        plt.draw()
        plt.pause(0.001)  # Pausiere für t, bevor der nächste Plot gezeigt wird

        profile_index = profile_index + 1

        if all_files_displayed:
            break               
            
except KeyboardInterrupt:
    print("process ended by user")

finally:
    plt.ioff()
    plt.show()