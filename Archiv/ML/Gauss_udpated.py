import json
from tqdm import tqdm
import GPy
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, Memory
from scipy.stats import norm
from sklearn.model_selection import KFold
import os

# Set up caching directory for memory (disable for testing if needed)
memory = Memory(location=r'B:/Gauss_caching', verbose=1)

# Directory containing CSV files
data_folder = r"B:\dataset_slicing\optimized_files\optimized_files"
all_file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]

# Check if file paths are accessible
print("Files to process:", len(all_file_paths))
for path in all_file_paths:
    print("Checking file:", path, "Exists:", os.path.exists(path))

# Initialize log file
log_file_path = r"B:/MA_Code/progress_log.json"
log_data = {}

# Function to load a specific row across multiple CSV files
#@memory.cache
def load_row_data(file_paths, row_index):
    """Loads the specified row index from each file in file_paths and ensures data is numeric."""
    row_data = []
    for file_path in file_paths:
        try:
            # Set the delimiter to ";" to ensure correct parsing
            df = pd.read_csv(file_path, delimiter=";")
            
            # Ensure only numeric data is loaded by converting or dropping non-numeric columns
            df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
            if row_index < len(df):  # Check if the row exists in the file
                row_data.append(df.iloc[row_index].values)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
    
    if row_data:
        return np.array(row_data)
    else:
        print(f"No data found for row index {row_index}")
        return None

# Function to optimize the kernel for Gaussian Process Regression
@memory.cache
def optimize_kernel(X_row, y_mean, length_scale_options=[0.1, 0.5, 1, 2, 5]):
    """Performs grid search to optimize RBF kernel length scale for GP regression."""
    best_kernel = None
    best_score = float('inf')
    for length_scale in length_scale_options:
        kernel = GPy.kern.RBF(input_dim=X_row.shape[1], lengthscale=length_scale)
        model = GPy.models.GPRegression(X_row, y_mean, kernel)
        model.optimize(max_iters=1000)
        score = -model.log_likelihood()
        if score < best_score:
            best_score = score
            best_kernel = kernel
    return best_kernel

# Function to train a Gaussian Process Regression model on a single row
#@memory.cache
def train_row_gpr(row_index, file_paths):
    """Trains a GP model on a specific row across the provided file paths."""
    X_row = load_row_data(file_paths, row_index)
    if X_row is None or X_row.size == 0:
        print(f"Empty data for row {row_index}")
        return None, None, None

    mean_height = X_row.mean(axis=0).reshape(-1, 1)
    variance = X_row.var(axis=0).reshape(-1, 1)

    best_kernel = optimize_kernel(X_row, mean_height)
    model = GPy.models.GPRegression(X_row, mean_height, best_kernel)
    model.optimize(max_iters=1000)

    return model, mean_height, variance

# Function to test the GP model on a new validation row
def test_row_gpr(row_index, validation_file_paths, model, mean, var, log_data, fold_idx):
    """Tests the GP model on a new row to identify deviations from satisfactory profiles."""
    new_row = load_row_data(validation_file_paths, row_index)
    if new_row is None or new_row.size == 0 or model is None:
        log_data[fold_idx]["rows"][row_index] = {
            "status": "Skipped (empty data or missing model)"
        }
        return row_index, None, None

    try:
        mean_pred, var_pred = model.predict(new_row)
        p_values = norm.cdf(abs(mean_pred - mean) / np.sqrt(var))
        flags = p_values < 0.05

        log_data[fold_idx]["rows"][row_index] = {
            "status": "Processed",
            "p_values": p_values.flatten().tolist(),
            "flags": flags.flatten().tolist()
        }
    except Exception as e:
        print(f"Error in test_row_gpr for row {row_index}: {e}")
        log_data[fold_idx]["rows"][row_index] = {
            "status": f"Error: {str(e)}"
        }
    
    return row_index, flags, mean_pred

# Number of profiles (rows) to analyze - reduced for testing
num_rows = 10  

# Set up K-Fold cross-validation with 2 splits for testing
kf = KFold(n_splits=2)

# Store results across all folds
cross_val_results = []

# Loop through each fold for training and validation
for fold_idx, (train_index, val_index) in enumerate(kf.split(all_file_paths), 1):
    print(f"Starting fold {fold_idx}")

    log_data[fold_idx] = {
        "status": "Processing",
        "rows": {}
    }

    train_files = [all_file_paths[i] for i in train_index]
    val_files = [all_file_paths[i] for i in val_index]

    # Train and test each row in parallel, with progress tracking
    results = Parallel(n_jobs=1)(  # Reduced n_jobs for memory and debug
        delayed(lambda row_idx: (
            train_row_gpr(row_idx, train_files),
            test_row_gpr(row_idx, val_files, *train_row_gpr(row_idx, train_files), log_data, fold_idx)
        ))(row_idx) for row_idx in tqdm(range(num_rows), desc=f"Processing fold {fold_idx}")
    )

    cross_val_results.append(results)
    log_data[fold_idx]["status"] = "Completed"

    with open(log_file_path, 'w') as log_file:
        json.dump(log_data, log_file, indent=4)

# Final log save
with open(log_file_path, 'w') as log_file:
    json.dump(log_data, log_file, indent=4)

print("Processing and logging completed.")
