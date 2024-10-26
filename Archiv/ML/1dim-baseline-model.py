import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import glob

# Define functions for calculating evaluation metrics
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Directory where CSV files are located
file_directory = r'B:\dataset_slicing\optimized_files'  # Replace with actual path

# Step 1: Load all CSV files and determine the minimum number of profiles (rows)
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

# Define lower and upper thresholds for valid height values
LOWER_THRESHOLD = 200
UPPER_THRESHOLD = 520

# Initialize lists to hold overall baseline model metrics
baseline_rmse_list, baseline_r2_list, baseline_mae_list = [], [], []

# Track excluded profiles for both training and validation
excluded_training_profiles = []
excluded_validation_profiles = []

# Iterate over each profile (row)
for profile_index in range(min_profiles):
    # Step 2: Calculate the mean height value for the profile across training files
    profile_data_train = [
        np.where(
            (train_file[profile_index, :] >= LOWER_THRESHOLD) & 
            (train_file[profile_index, :] <= UPPER_THRESHOLD),
            train_file[profile_index, :],
            np.nan
        )
        for train_file in train_data
    ]

    # Check if all values are NaN in profile_data_train
    if np.isnan(profile_data_train).all():
        excluded_training_profiles.append(profile_index + 1)
        continue

    # Calculate the mean profile for the current profile index, ignoring NaNs
    baseline_profile = np.nanmean(profile_data_train, axis=0)
    
    # Initialize lists to store profile-level metrics across validation files
    profile_rmse, profile_r2, profile_mae = [], [], []
    profile_excluded_from_validation = True  # Flag to track if all validation sets are excluded for this profile

    # Step 3: Apply baseline prediction to each profile in validation data
    for val_file in val_data:
        # Get actual values for the current profile in the validation file, applying thresholds
        val_profile = np.where(
            (val_file[profile_index, :] >= LOWER_THRESHOLD) & 
            (val_file[profile_index, :] <= UPPER_THRESHOLD),
            val_file[profile_index, :],
            np.nan
        )

        # Filter valid indices (non-NaN) to ensure proper metric calculation
        valid_idx = ~np.isnan(val_profile) & ~np.isnan(baseline_profile)
        if valid_idx.sum() < 2:  # Skip if less than two valid points
            print(f"Validation for profile {profile_index + 1} skipped - insufficient valid values.")
            continue

        # If at least one validation set is processed, set the flag to False
        profile_excluded_from_validation = False

        # Calculate metrics between the actual and baseline values for valid indices
        rmse = root_mean_squared_error(val_profile[valid_idx], baseline_profile[valid_idx])
        r2_value = r2(val_profile[valid_idx], baseline_profile[valid_idx])
        mae_value = mae(val_profile[valid_idx], baseline_profile[valid_idx])

        profile_rmse.append(rmse)
        profile_r2.append(r2_value)
        profile_mae.append(mae_value)

    # Aggregate metrics for the profile across validation files
    if profile_rmse:
        baseline_rmse_list.append(np.mean(profile_rmse))
        baseline_r2_list.append(np.mean(profile_r2))
        baseline_mae_list.append(np.mean(profile_mae))
    
    # If no valid validation data, mark the profile as excluded from validation
    if profile_excluded_from_validation:
        excluded_validation_profiles.append(profile_index + 1)

    # Plot the baseline and validation profiles for the current profile
    plt.figure(figsize=(10, 5))
    plt.plot(baseline_profile, label="Baseline Prediction", color="red", linestyle="--")
    for i, val_file in enumerate(val_data):
        val_profile = np.where(
            (val_file[profile_index, :] >= LOWER_THRESHOLD) & 
            (val_file[profile_index, :] <= UPPER_THRESHOLD),
            val_file[profile_index, :],
            np.nan
        )
        plt.plot(val_profile, label=f"Validation File {i+1}", alpha=0.5)

    plt.title(f"Profile {profile_index + 1} - Baseline Prediction vs Validation Data")
    plt.xlabel("Position along Profile")
    plt.ylabel("Height Value")
    plt.legend()
    plt.show()

# Step 4: Calculate overall baseline metrics across all profiles
if baseline_rmse_list:
    overall_rmse = np.mean(baseline_rmse_list)
    overall_r2 = np.mean(baseline_r2_list)
    overall_mae = np.mean(baseline_mae_list)

    print("\nBaseline Model - Overall Results:")
    print(f"Average RMSE: {overall_rmse:.4f}")
    print(f"Average R²: {overall_r2:.4f}")
    print(f"Average MAE: {overall_mae:.4f}")
else:
    print("No valid profiles to calculate baseline metrics.")

# Output the excluded profiles for training and validation
print(f"\nTotal profiles excluded from training: {len(excluded_training_profiles)}")
print(f"Profiles excluded from training: {excluded_training_profiles}")
print(f"Total profiles excluded from validation: {len(excluded_validation_profiles)}")
print(f"Profiles excluded from validation: {excluded_validation_profiles}")
