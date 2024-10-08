# Import necessary libraries
import GPy
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.model_selection import KFold

# Function to load a specific row across multiple CSV files
def load_row_data(file_paths, row_index):
    row_data = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        if row_index < len(df):  # Check if row exists
            row_data.append(df.iloc[row_index].values)
    return np.array(row_data)

# Function to train a GP model on a single row (combined across all training files)
def train_row_gpr(row_index, file_paths, kernel=None):
    X_row = load_row_data(file_paths, row_index)
    
    if X_row.size == 0:
        return None, None, None
    
    # Calculate mean and variance of the satisfactory profiles
    mean_height = X_row.mean(axis=0).reshape(-1, 1)
    variance = X_row.var(axis=0).reshape(-1, 1)
    
    # Train the GP model for this row
    if kernel is None:
        kernel = GPy.kern.RBF(input_dim=X_row.shape[1])
    
    model = GPy.models.GPRegression(X_row, mean_height, kernel)
    model.optimize(max_iters=1000)
    
    return model, mean_height, variance

# Function to test the trained GP model on a validation row
def test_row_gpr(row_index, validation_file_paths, model, mean, var):
    new_row = load_row_data(validation_file_paths, row_index)
    
    if new_row.size == 0 or model is None:
        return row_index, None, None

    # Predict using the trained GP model
    mean_pred, var_pred = model.predict(new_row)
    
    # Calculate p-values to evaluate deviation from satisfactory profiles
    p_values = norm.cdf(abs(mean_pred - mean) / np.sqrt(var))
    flags = p_values < 0.05  # Flag profiles with p-value < 0.05 as deviations
    
    return row_index, flags, mean_pred

# Paths for all profiles
all_file_paths = ["path_to_file1.csv", "path_to_file2.csv", ..., "path_to_file30.csv"]
num_rows = 5000  # Assuming we have 5,000 profiles (rows)

# Kernel for GPR
kernel = GPy.kern.RBF(input_dim=1)

# Create a KFold object for 6-fold cross-validation
kf = KFold(n_splits=6)

# Results from all folds
cross_val_results = []

# Loop over each fold
for fold, (train_index, val_index) in enumerate(kf.split(all_file_paths), 1):
    print(f"Starting fold {fold}")
    
    # Define training and validation file sets for this fold
    train_files = [all_file_paths[i] for i in train_index]
    val_files = [all_file_paths[i] for i in val_index]
    
    # Train and test each row across files in parallel
    results = Parallel(n_jobs=-1)(
        delayed(
            lambda row_idx: (
                # Train model on training files for this row
                *train_row_gpr(row_idx, train_files, kernel),
                # Test this row on validation files
                test_row_gpr(row_idx, val_files, *train_row_gpr(row_idx, train_files, kernel))
            )
        )(row_idx) for row_idx in range(num_rows)
    )
    
    # Collect results from this fold
    cross_val_results.append(results)

# Process and display flagged profiles for all folds
for fold_idx, fold_results in enumerate(cross_val_results, 1):
    print(f"Results for fold {fold_idx}:")
    for row_index, (flags, mean_pred) in fold_results:
        if flags is not None:
            flagged_profiles = np.where(flags)[0]
            print(f"Row {row_index} flagged profiles (likely defective):", flagged_profiles)
