import pandas as pd
import numpy as np
import GPy
import glob
import matplotlib.pyplot as plt
import os
from py_helpers import select_folder

#select directory
directory = select_folder()
training_files = glob.glob(os.path.join(directory, "*.csv"))

#list for input data
X = [] 
#list for output data
Y = []

#read data from each csv file for training
for file in training_files:
    try:
        print(f"reading file {file}")
        df = pd.read_csv(file, header=None, sep=";", dtype="int16")

        #df.iterrows() is a generator that yields index and row data for each row in the DataFrame 
        # -> (index, Series), where i is the row index and profile contains the data of the corresponding index
        # Iterate over each profile (row) in the DataFrame
        for i, profile in df.iterrows():
            # column vector necessary for gaussian process regression
            # Create a column vector with the profile index for each height value = give each height value of row the index i (1 for first row and so on),
            # creating a column vector (array with one column), where each element is the profile index 
            # means for first row, that there is an array with one column where each element is [1]
                #np.full(shape, fill_value):
                    #np -> numPy lib
                    #full array of given shape and type, filled with values
                    #shape: tuple that defines shape of array in this case ("len(profile)") rows and 1 column
                    #fill value: in this case "i", which is row index of df

            profile_index = np.full((len(profile), 1), i, dtype=np.int16)
            # Extract the height values and reshape them into a column vector to make it compatible with GPR
                #access underlying data of "profile" series as numpy array -> height values for current profile
                #.reshape(-1, 1): 
                    #-1 specific value in numpy, tells automatically to determine the no. of rows based on length of array | 
                    # 1 specifies that array should have one column 
            height_values = profile.values.reshape(-1, 1).astype(np.int16)

            # Print shapes and types for debugging
            #print(f"Profile index shape: {profile_index.shape}, dtype: {profile_index.dtype}")
            #print(f"Height values shape: {height_values.shape}, dtype: {height_values.dtype}")

            # Append the profile indices to the input data list
            X.append(profile_index)
            # Append the height values to the output data list
            Y.append(height_values)

    except Exception as error:
        print(f"error processing file {file}: {error}")
    
# Convert lists to numpy arrays for training: convert list of arrays 
try:
    X = np.vstack(X).astype(np.int16)
    Y = np.vstack(Y).astype(np.int16)
except Exception as e:
    print(f"Error converting lists to numpy arrays: {e}")

# Print shapes of the arrays for verification
print("Shape of X (input data):", X.shape)
print("Shape of Y (output data):", Y.shape)

# Define and train the Gaussian Process model
try:
    kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
    model = GPy.models.GPRegression(X, Y, kernel)
    model.optimize(messages=True)
except Exception as e:
    print(f"error during model training: {e}")

# Visualize the trained model
plt.figure(figsize=(10, 6))
model.plot()
plt.xlabel('Profile Index')
plt.ylabel('Height Values')
plt.title('Trained Gaussian Process Regression Model')
plt.show()


    

