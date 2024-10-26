import pandas as pd
import numpy as np
import GPy
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import glob
from joblib import Parallel, delayed
from tqdm import tqdm

# Function for calculating RMSE
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Directory where CSV files are located
file_directory = r'B:\dataset_slicing\optimized_files'  # Replace with actual path

# Load all CSV files and determine the minimum number of profiles (rows)
file_paths = glob.glob(f"{file_directory}/*.csv")
all_measurements = []
min_profiles = float('inf')

for file_path in file_paths:
    df = pd.read_csv(file_path, sep=";", header=None)
    min_profiles = min(min_profiles, df.shape[0])
    all_measurements.append(df)

# Ensure all files have the same number of profiles by truncating to the minimum
all_measurements = [df.iloc[:min_profiles, :].values for df in all_measurements]

# Split data into training (80%) and validation (20%)
train_size = int(0.8 * len(all_measurements))
train_data = all_measurements[:train_size]
val_data = all_measurements[train_size:]

# Set lower and upper bounds for acceptable height values
LOWER_THRESHOLD = 200
UPPER_THRESHOLD = 520

# Function to process each profile with GPR
def process_profile(profile_index):
    # Initialize metrics storage
    profile_rmse, profile_r2, profile_mae = [], [], []
    profile_excluded_from_validation = True  # Flag for valid validation check
    
    # Step 1: Filter and calculate mean profile across training files
    profile_data_train = [
        np.where(
            (train_file[profile_index, :] >= LOWER_THRESHOLD) & 
            (train_file[profile_index, :] <= UPPER_THRESHOLD),
            train_file[profile_index, :],
            np.nan
        )
        for train_file in train_data
    ]

    if np.isnan(profile_data_train).all():
        return None, None, profile_index + 1, None  # Mark as excluded from training

    profile_data_train = np.array(profile_data_train)
    Y_train = np.nanmean(profile_data_train, axis=0).reshape(-1, 1)
    X = np.arange(Y_train.shape[0]).reshape(-1, 1)

    # Define and train the GPR model
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=20.)
    model = GPy.models.GPRegression(X, Y_train, kernel)
    model.optimize(messages=False)

    # Step 2: Apply GPR prediction on validation data and calculate metrics
    for val_file in val_data:
        val_profile = np.where(
            (val_file[profile_index, :] >= LOWER_THRESHOLD) & 
            (val_file[profile_index, :] <= UPPER_THRESHOLD),
            val_file[profile_index, :],
            np.nan
        ).reshape(-1, 1)

        valid_idx = ~np.isnan(val_profile.flatten())
        if valid_idx.sum() < 2:
            continue

        Y_pred, _ = model.predict(X)
        Y_pred = Y_pred.flatten()
        val_profile = val_profile.flatten()
        valid_combined_idx = valid_idx & ~np.isnan(Y_pred)

        if valid_combined_idx.sum() < 2:
            continue

        profile_rmse.append(root_mean_squared_error(val_profile[valid_combined_idx], Y_pred[valid_combined_idx]))
        profile_r2.append(r2(val_profile[valid_combined_idx], Y_pred[valid_combined_idx]))
        profile_mae.append(mae(val_profile[valid_combined_idx], Y_pred[valid_combined_idx]))
        profile_excluded_from_validation = False

    # If no valid validation data, mark the profile as excluded
    if profile_excluded_from_validation:
        return None, None, None, profile_index + 1
    
    # Calculate average metrics across validation files for this profile
    avg_rmse = np.mean(profile_rmse) if profile_rmse else None
    avg_r2 = np.mean(profile_r2) if profile_r2 else None
    avg_mae = np.mean(profile_mae) if profile_mae else None
    return avg_rmse, avg_r2, avg_mae, None

# Run profile processing in parallel
results = Parallel(n_jobs=-1)(delayed(process_profile)(i) for i in tqdm(range(min_profiles), desc="Processing Profiles"))

# Aggregate metrics and track exclusions
gpr_rmse_list, gpr_r2_list, gpr_mae_list = [], [], []
excluded_training_profiles, excluded_validation_profiles = [], []

for avg_rmse, avg_r2, avg_mae, excluded_profile in results:
    if excluded_profile is not None:
        if excluded_profile in excluded_training_profiles:
            excluded_training_profiles.append(excluded_profile)
        else:
            excluded_validation_profiles.append(excluded_profile)
    else:
        if avg_rmse is not None:
            gpr_rmse_list.append(avg_rmse)
            gpr_r2_list.append(avg_r2)
            gpr_mae_list.append(avg_mae)

# Step 3: Calculate overall GPR metrics
if gpr_rmse_list:
    overall_rmse = np.mean(gpr_rmse_list)
    overall_r2 = np.mean(gpr_r2_list)
    overall_mae = np.mean(gpr_mae_list)

    print("\nGPR Model - Overall Results:")
    print(f"Average RMSE: {overall_rmse:.4f}")
    print(f"Average R²: {overall_r2:.4f}")
    print(f"Average MAE: {overall_mae:.4f}")
else:
    print("No valid profiles to calculate GPR metrics.")

# Output the excluded profiles for training and validation
print(f"\nTotal profiles excluded from training: {len(excluded_training_profiles)}")
print(f"Profiles excluded from training: {excluded_training_profiles}")
print(f"Total profiles excluded from validation: {len(excluded_validation_profiles)}")
print(f"Profiles excluded from validation: {excluded_validation_profiles}")
