import pandas as pd
import numpy as np
import GPy
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
from joblib import Parallel, delayed
from tqdm import tqdm
import glob
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

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

# Directory where CSV files are located
file_directory = r'B:\dataset_slicing\optimized_files'  # Replace with your actual path

# Set lower and upper bounds for acceptable height values
LOWER_THRESHOLD = 200
UPPER_THRESHOLD = 520

# Load data
all_measurements, min_profiles = load_data(file_directory)

# Split data into training (80%) and validation (20%)
train_size = int(0.8 * len(all_measurements))
train_data = all_measurements[:train_size]
val_data = all_measurements[train_size:]

# Function to process each profile with GPR
def process_profile_gpr(profile_index):
    # Initialize metrics storage
    profile_rmse, profile_r2, profile_mae = [], [], []
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
        return (None, None, None, None, profile_excluded_from_training,
                profile_excluded_from_validation, profile_index + 1,
                profile_excluded_from_baseline_training, profile_excluded_from_baseline_validation)

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
        return (None, None, None, None, profile_excluded_from_training,
                profile_excluded_from_validation, profile_index + 1,
                profile_excluded_from_baseline_training, profile_excluded_from_baseline_validation)

    # Scale X and Y_train
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_valid)
    Y_scaled = scaler_Y.fit_transform(Y_train_valid)

    # Define and train the GPR model using only the RBF kernel
    kernel = GPy.kern.Matern52(input_dim=1)
    model = GPy.models.GPRegression(X_scaled, Y_scaled, kernel)
    model.optimize(messages=False)

    # Prepare full prediction array
    full_Y_pred = np.full(len(Y_train), np.nan)
    Y_pred_scaled, _ = model.predict(X_scaled)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    full_Y_pred[valid_train_idx] = Y_pred.flatten()

    # Initialize dictionaries to store residuals with indices
    residuals_gpr_dict = {}
    residuals_baseline_dict = {}

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

        # Check if baseline_valid has NaNs
        if np.isnan(baseline_valid).all():
            profile_excluded_from_baseline_validation = True
            continue

        # Calculate metrics for this validation file
        rmse_gpr, mae_gpr, r2_gpr = calculate_metrics(val_profile_valid, pred_valid)
        profile_rmse.append(rmse_gpr)
        profile_r2.append(r2_gpr)
        profile_mae.append(mae_gpr)

        # Collect residuals with indices
        for idx, res_gpr, res_baseline in zip(indices_valid, val_profile_valid - pred_valid, val_profile_valid - baseline_valid):
            residuals_gpr_dict[(profile_index, idx)] = res_gpr
            residuals_baseline_dict[(profile_index, idx)] = res_baseline

        profile_excluded_from_validation = False  # Found valid data
        profile_excluded_from_baseline_validation = False

    # If no valid validation data, mark the profile as excluded
    if profile_excluded_from_validation:
        return (None, None, None, None, False, True, profile_index + 1,
                profile_excluded_from_baseline_training, profile_excluded_from_baseline_validation)

    # Calculate average metrics across validation files for this profile
    avg_rmse = np.mean(profile_rmse) if profile_rmse else None
    avg_r2 = np.mean(profile_r2) if profile_r2 else None
    avg_mae = np.mean(profile_mae) if profile_mae else None

    return (avg_rmse, avg_r2, avg_mae, (residuals_gpr_dict, residuals_baseline_dict),
            profile_excluded_from_training, profile_excluded_from_validation, profile_index + 1,
            profile_excluded_from_baseline_training, profile_excluded_from_baseline_validation)

# Run GPR profile processing in parallel
results_gpr = Parallel(n_jobs=-1)(
    delayed(process_profile_gpr)(i) for i in tqdm(range(min_profiles), desc="Processing GPR Profiles")
)

# Aggregate metrics, residuals, and track exclusions
gpr_rmse_list, gpr_r2_list, gpr_mae_list = [], [], []
all_residuals_gpr_dict = {}
all_residuals_baseline_dict = {}
excluded_training_profiles_gpr, excluded_validation_profiles_gpr = [], []
excluded_training_profiles_baseline, excluded_validation_profiles_baseline = [], []

for (avg_rmse, avg_r2, avg_mae, residuals,
     excluded_from_training, excluded_from_validation, profile_index,
     excluded_from_baseline_training, excluded_from_baseline_validation) in results_gpr:
    if excluded_from_training:
        excluded_training_profiles_gpr.append(profile_index)
    elif excluded_from_validation:
        excluded_validation_profiles_gpr.append(profile_index)
    else:
        if avg_rmse is not None:
            gpr_rmse_list.append(avg_rmse)
            gpr_r2_list.append(avg_r2)
            gpr_mae_list.append(avg_mae)
            # Merge residuals dictionaries
            residuals_gpr_dict, residuals_baseline_dict = residuals
            all_residuals_gpr_dict.update(residuals_gpr_dict)
            all_residuals_baseline_dict.update(residuals_baseline_dict)

    # Collect exclusions for the baseline model
    if excluded_from_baseline_training:
        excluded_training_profiles_baseline.append(profile_index)
    if excluded_from_baseline_validation:
        excluded_validation_profiles_baseline.append(profile_index)

# Align residuals based on indices
common_indices = set(all_residuals_gpr_dict.keys()) & set(all_residuals_baseline_dict.keys())

# Extract aligned residuals
aligned_residuals_gpr = np.array([all_residuals_gpr_dict[idx] for idx in common_indices])
aligned_residuals_baseline = np.array([all_residuals_baseline_dict[idx] for idx in common_indices])

# Perform paired t-test with detailed calculations
if len(aligned_residuals_gpr) > 1 and len(aligned_residuals_baseline) > 1:
    # Calculate the differences
    differences = aligned_residuals_baseline - aligned_residuals_gpr

    # Sample size
    n = len(differences)

    # Means and standard deviations
    mean_residual_baseline = np.mean(aligned_residuals_baseline)
    std_residual_baseline = np.std(aligned_residuals_baseline, ddof=1)
    mean_residual_gpr = np.mean(aligned_residuals_gpr)
    std_residual_gpr = np.std(aligned_residuals_gpr, ddof=1)

    # Mean and std of differences
    mean_difference = np.mean(differences)
    std_difference = np.std(differences, ddof=1)

    # Standard error
    standard_error = std_difference / np.sqrt(n)

    # Manually calculate t-statistic
    t_statistic_manual = mean_difference / standard_error

    # Degrees of freedom
    df = n - 1

    # p-value from t-statistic
    from scipy.stats import t
    p_value_manual = 2 * t.sf(np.abs(t_statistic_manual), df)

    # Using scipy's ttest_rel for comparison
    t_statistic_scipy, p_value_scipy = ttest_rel(aligned_residuals_baseline, aligned_residuals_gpr)

else:
    t_statistic_manual = p_value_manual = None
    t_statistic_scipy = p_value_scipy = None
    n = 0

# Calculate overall GPR metrics
if gpr_rmse_list:
    overall_rmse_gpr = np.mean(gpr_rmse_list)
    overall_r2_gpr = np.mean(gpr_r2_list)
    overall_mae_gpr = np.mean(gpr_mae_list)
else:
    overall_rmse_gpr = overall_r2_gpr = overall_mae_gpr = None

# Output the results with detailed print statements
print("\nGPR Model - Overall Results:")
if overall_rmse_gpr is not None:
    print(f"Average RMSE: {overall_rmse_gpr:.4f}")
    print(f"Average R²: {overall_r2_gpr:.4f}")
    print(f"Average MAE: {overall_mae_gpr:.4f}")
else:
    print("No valid profiles to calculate GPR metrics.")

if t_statistic_manual is not None:
    print("\nPaired t-test between Baseline and GPR residuals:")
    print(f"Sample size (n): {n}")
    print(f"Baseline residuals - Mean: {mean_residual_baseline:.4f}, Std Dev: {std_residual_baseline:.4f}")
    print(f"GPR residuals - Mean: {mean_residual_gpr:.4f}, Std Dev: {std_residual_gpr:.4f}")
    print(f"Differences (Baseline - GPR) - Mean: {mean_difference:.4f}, Std Dev: {std_difference:.4f}")
    print(f"Standard Error of Differences: {standard_error:.4f}")
    print(f"Degrees of Freedom: {df}")
    print(f"Manually Calculated t-statistic: {t_statistic_manual:.4f}")
    print(f"Manually Calculated p-value: {p_value_manual:.4f}")
    print(f"t-statistic from scipy: {t_statistic_scipy:.4f}")
    print(f"p-value from scipy: {p_value_scipy:.4f}")
    significance = "Significant difference" if p_value_scipy < 0.05 else "No significant difference"
    print(f"Statistical significance: {significance}")
else:
    print("Not enough residuals to perform paired t-test.")

# Output the excluded profiles
print(f"\nTotal profiles: {min_profiles}")
print(f"Total profiles excluded from GPR training: {len(excluded_training_profiles_gpr)}")
print(f"Profiles excluded from GPR training: {excluded_training_profiles_gpr}")
print(f"Total profiles excluded from GPR validation: {len(excluded_validation_profiles_gpr)}")
print(f"Profiles excluded from GPR validation: {excluded_validation_profiles_gpr}")

print(f"\nTotal profiles excluded from Baseline training: {len(excluded_training_profiles_baseline)}")
print(f"Profiles excluded from Baseline training: {excluded_training_profiles_baseline}")
print(f"Total profiles excluded from Baseline validation: {len(excluded_validation_profiles_baseline)}")
print(f"Profiles excluded from Baseline validation: {excluded_validation_profiles_baseline}")
