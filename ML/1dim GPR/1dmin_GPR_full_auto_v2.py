import pandas as pd
import numpy as np
import GPy
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from tqdm import tqdm
import glob
import warnings
import json
import os
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Plotting flag
PLOT_PROFILES = True  # Set to True to enable plotting

# Functions for calculating metrics
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# Function to load and preprocess data
def load_data(file_directory):
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

    return all_measurements, min_profiles

# Plotting function for each profile
def plot_profile(X, Y_true, baseline_pred, Y_gpr_pred, Y_gpr_std, profile_index):
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y_true, label="Actual Values", color="blue")
    plt.plot(X, baseline_pred, label="Baseline Prediction", linestyle="--", color="green")
    plt.plot(X, Y_gpr_pred, label="GPR Prediction", color="red")
    plt.fill_between(
        X.flatten(),
        (Y_gpr_pred.flatten() - 1.96 * Y_gpr_std),
        (Y_gpr_pred.flatten() + 1.96 * Y_gpr_std),
        color="orange", alpha=0.2, label="95% Confidence Interval (GPR)"
    )
    plt.title(f"Profile {profile_index}: Baseline vs GPR Predictions")
    plt.xlabel("Position along Profile")
    plt.ylabel("Height")
    plt.legend()
    plt.draw()
    plt.pause(2)   # Display the plot for 2 seconds
    plt.close()    # Close the plot before moving to the next one

# Directory where CSV files are located
file_directory = r'B:\dataset_slicing\optimized_files\optimized_files'  # Replace with your actual path

# Set lower and upper bounds for acceptable height values
LOWER_THRESHOLD = 200
UPPER_THRESHOLD = 520

# Load data
all_measurements, min_profiles = load_data(file_directory)

# Set up KFold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Initialize lists to aggregate results across folds
gpr_rmse_list_all_folds = []
gpr_r2_list_all_folds = []
gpr_mae_list_all_folds = []
all_residuals_gpr_dict_all_folds = {}
all_residuals_baseline_dict_all_folds = {}
profile_p_values = {}  # Store p-values for each profile

# Store metrics for each profile in JSON format
profile_results = {}

# Initialize lists to store mean residuals
mean_residuals_gpr_all_profiles = []
mean_residuals_baseline_all_profiles = []

# Function to process each profile with GPR
def process_profile_gpr(profile_index, train_data, val_data):
    # Initialize metrics storage for GPR and baseline
    profile_rmse_gpr, profile_r2_gpr, profile_mae_gpr = [], [], []
    profile_rmse_baseline, profile_r2_baseline, profile_mae_baseline = [], [], []
    residuals_gpr_list, residuals_baseline_list = [], []
    mean_residual_gpr, mean_residual_baseline = None, None
    
    profile_excluded_from_training = False
    profile_excluded_from_validation = False
    profile_excluded_from_baseline_training = False
    profile_excluded_from_baseline_validation = False

    # Step 1: Filter and calculate mean profile across training files
    profile_data_train = []
    for train_file in train_data:
        profile = np.where(
            (train_file[profile_index, :] >= LOWER_THRESHOLD) &
            (train_file[profile_index, :] <= UPPER_THRESHOLD),
            train_file[profile_index, :],
            np.nan
        )
        profile_data_train.append(profile)

    profile_data_train = np.array(profile_data_train)

    # Check if all training data is NaN after filtering
    if np.isnan(profile_data_train).all():
        profile_excluded_from_training = True
        profile_excluded_from_baseline_training = True
        return (None, None, None, None, None, None, profile_excluded_from_training,
                profile_excluded_from_validation, profile_index + 1,
                profile_excluded_from_baseline_training, profile_excluded_from_baseline_validation,
                None, None)  # No plotting data

    # Compute baseline profile by taking the mean across training files, ignoring NaNs
    baseline_profile = np.nanmean(profile_data_train, axis=0)
    Y_train = baseline_profile.copy()
    X = np.arange(Y_train.shape[0])

    # Remove NaN values from training data
    valid_train_idx = ~np.isnan(Y_train)
    X_valid = X[valid_train_idx].reshape(-1, 1)
    Y_train_valid = Y_train[valid_train_idx].reshape(-1, 1)

    if len(Y_train_valid) < 2:
        profile_excluded_from_training = True
        profile_excluded_from_baseline_training = True
        return (None, None, None, None, None, None, profile_excluded_from_training,
                profile_excluded_from_validation, profile_index + 1,
                profile_excluded_from_baseline_training, profile_excluded_from_baseline_validation,
                None, None)  # No plotting data

    # Scale X and Y_train
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_valid)
    Y_scaled = scaler_Y.fit_transform(Y_train_valid)

    # Define and train the GPR model using the RBF kernel
    kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=3)
    model = GPy.models.GPRegression(X_scaled, Y_scaled, kernel)
    model.optimize(messages=False)

    # Prepare full prediction array and compute standard deviations
    full_Y_pred = np.full(len(Y_train), np.nan)
    full_Y_std = np.full(len(Y_train), np.nan)
    Y_pred_scaled, Y_var_scaled = model.predict(X_scaled)
    Y_std_scaled = np.sqrt(Y_var_scaled)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    Y_std = scaler_Y.scale_ * Y_std_scaled.flatten()
    full_Y_pred[valid_train_idx] = Y_pred.flatten()
    full_Y_std[valid_train_idx] = Y_std

    # Step 2: Apply GPR prediction on validation data and collect residuals
    profile_excluded_from_validation = True  # Assume excluded until valid data is found
    profile_excluded_from_baseline_validation = True

    for val_file in val_data:
        val_profile = np.where(
            (val_file[profile_index, :] >= LOWER_THRESHOLD) &
            (val_file[profile_index, :] <= UPPER_THRESHOLD),
            val_file[profile_index, :],
            np.nan
        )

        # Valid indices where both val_profile and predictions are not NaN
        valid_idx = ~np.isnan(val_profile) & ~np.isnan(full_Y_pred)

        if valid_idx.sum() < 2:
            continue

        val_profile_valid = val_profile[valid_idx]
        pred_valid = full_Y_pred[valid_idx]
        baseline_valid = Y_train[valid_idx]
        indices_valid = np.where(valid_idx)[0]
        std_valid = full_Y_std[valid_idx]

        # Check if baseline_valid has NaNs
        if np.isnan(baseline_valid).all():
            profile_excluded_from_baseline_validation = True
            continue

        # Calculate GPR metrics for this validation file
        rmse_gpr, mae_gpr, r2_gpr = calculate_metrics(val_profile_valid, pred_valid)
        profile_rmse_gpr.append(rmse_gpr)
        profile_r2_gpr.append(r2_gpr)
        profile_mae_gpr.append(mae_gpr)

        # Calculate baseline metrics for this validation file
        rmse_baseline, mae_baseline, r2_baseline = calculate_metrics(val_profile_valid, baseline_valid)
        profile_rmse_baseline.append(rmse_baseline)
        profile_r2_baseline.append(r2_baseline)
        profile_mae_baseline.append(mae_baseline)

        # Collect residuals
        residuals_gpr = val_profile_valid - pred_valid
        residuals_baseline = val_profile_valid - baseline_valid
        residuals_gpr_list.extend(residuals_gpr)
        residuals_baseline_list.extend(residuals_baseline)

        profile_excluded_from_validation = False  # Found valid data
        
        # Calculate mean residuals for GPR and baseline
        mean_residual_gpr = np.mean(residuals_gpr) if len(residuals_gpr) > 0 else None
        mean_residual_baseline = np.mean(residuals_baseline) if len(residuals_baseline) > 0 else None
        profile_excluded_from_baseline_validation = False

    # If no valid validation data, mark the profile as excluded
    if profile_excluded_from_validation:
        return (None, None, None, None, None, None, False, True, profile_index + 1,
                profile_excluded_from_baseline_training, profile_excluded_from_baseline_validation,
                None, None)  # No plotting data

    # Calculate average metrics across validation files for this profile
    avg_rmse_gpr = np.mean(profile_rmse_gpr) if profile_rmse_gpr else None
    avg_r2_gpr = np.mean(profile_r2_gpr) if profile_r2_gpr else None
    avg_mae_gpr = np.mean(profile_mae_gpr) if profile_mae_gpr else None

    avg_rmse_baseline = np.mean(profile_rmse_baseline) if profile_rmse_baseline else None
    avg_r2_baseline = np.mean(profile_r2_baseline) if profile_r2_baseline else None
    avg_mae_baseline = np.mean(profile_mae_baseline) if profile_mae_baseline else None

    # Calculate p-value using paired t-test
    if residuals_gpr_list and residuals_baseline_list:
        t_stat, p_value = ttest_rel(residuals_gpr_list, residuals_baseline_list)
    else:
        p_value = None

    return (avg_rmse_gpr, avg_r2_gpr, avg_mae_gpr,
            avg_rmse_baseline, avg_r2_baseline, avg_mae_baseline,
            profile_excluded_from_training, profile_excluded_from_validation, profile_index + 1,
            profile_excluded_from_baseline_training, profile_excluded_from_baseline_validation,
            residuals_gpr_list, p_value, mean_residual_gpr, mean_residual_baseline)  # Return residuals and p-value

# Iterate over each fold
for fold_index, (train_indices, val_indices) in enumerate(kf.split(all_measurements)):
    print(f"\nProcessing Fold {fold_index+1}/{kf.get_n_splits()}")
    train_data = [all_measurements[i] for i in train_indices]
    val_data = [all_measurements[i] for i in val_indices]

    # Run GPR profile processing in parallel for this fold
    results_gpr = Parallel(n_jobs=-1)(
        delayed(process_profile_gpr)(i, train_data, val_data) for i in tqdm(range(min_profiles), desc=f"Processing Profiles for Fold {fold_index+1}")
    )

    # Collect results for each profile and save to JSON format
    
    for (avg_rmse_gpr, avg_r2_gpr, avg_mae_gpr,
        avg_rmse_baseline, avg_r2_baseline, avg_mae_baseline,
        excluded_from_training, excluded_from_validation, profile_index,
        excluded_from_baseline_training, excluded_from_baseline_validation,
        residuals_gpr_list, p_value, mean_residual_gpr, mean_residual_baseline) in results_gpr:
        
        # Ensure a result entry for each profile, even if excluded
        profile_key = f"Fold_{fold_index + 1}_Profile_{profile_index}"
        
        # Create entry with calculated metrics or None if excluded
        profile_results[profile_key] = {
            "Fold": fold_index + 1,
            "Baseline RMSE": avg_rmse_baseline if avg_rmse_baseline is not None else None,
            "Baseline MAE": avg_mae_baseline if avg_mae_baseline is not None else None,
            "Baseline R^2": avg_r2_baseline if avg_r2_baseline is not None else None,
            "Excluded from Training": excluded_from_training,
            "Excluded from Validation": excluded_from_validation,
            "Excluded from Baseline Training": excluded_from_baseline_training,
            "Excluded from Baseline Validation": excluded_from_baseline_validation,
            "GPR RMSE": avg_rmse_gpr if avg_rmse_gpr is not None else None,
            "GPR MAE": avg_mae_gpr if avg_mae_gpr is not None else None,
            "GPR R^2": avg_r2_gpr if avg_r2_gpr is not None else None,
            "P-value": p_value
        }

# Save the results to a JSON file
output_file = os.path.join(file_directory, "profile_metrics.json")
with open(output_file, "w") as f:
    json.dump(profile_results, f, indent=4)

# Calculate mean residuals across all profiles
mean_residual_gpr_overall = np.mean(mean_residuals_gpr_all_profiles) if mean_residuals_gpr_all_profiles else None
mean_residual_baseline_overall = np.mean(mean_residuals_baseline_all_profiles) if mean_residuals_baseline_all_profiles else None

# Add mean residuals to the JSON output
profile_results["Overall_Mean_Residuals"] = {
    "GPR": mean_residual_gpr_overall,
    "Baseline": mean_residual_baseline_overall
}

print(f"\nResults for each profile have been saved to {output_file}")
