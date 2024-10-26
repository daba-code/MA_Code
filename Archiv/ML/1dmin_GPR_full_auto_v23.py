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

    return all_measurements, min_profiles, file_paths

def plot_profile(X, Y_true, baseline_pred, Y_gpr_pred, Y_gpr_std, profile_index, fold_index, val_file_idx, file_directory, val_file_name):
    # Filter X and Y_true to match the valid data length
    X_valid = X[~np.isnan(Y_true)]
    Y_true_valid = Y_true[~np.isnan(Y_true)]

    # Ensure all arrays have the same length
    if len(X_valid) == len(Y_true_valid) == len(baseline_pred) == len(Y_gpr_pred) == len(Y_gpr_std):
        plt.figure(figsize=(10, 6))
        plt.plot(X_valid, Y_true_valid, label="Actual Values", color="blue")
        plt.plot(X_valid, baseline_pred, label="Baseline Prediction", linestyle="--", color="green")
        plt.plot(X_valid, Y_gpr_pred, label="GPR Prediction", color="red")
        plt.fill_between(
            X_valid.flatten(),
            (Y_gpr_pred.flatten() - 1.96 * Y_gpr_std),
            (Y_gpr_pred.flatten() + 1.96 * Y_gpr_std),
            color="orange", alpha=0.2, label="95% Confidence Interval (GPR)"
        )
        
        # Use the file name in the plot title
        clean_val_file_name = os.path.splitext(os.path.basename(val_file_name))[0]
        plt.title(f"Fold {fold_index + 1} - {clean_val_file_name} - Profile {profile_index + 1}: Baseline vs GPR Predictions")
        plt.xlabel("Position along Profile")
        plt.ylabel("Height")
        plt.legend()

        # Create output folder if it doesn't exist
        output_folder = os.path.join(file_directory, "plots")
        os.makedirs(output_folder, exist_ok=True)

        # Save the plot with a filename that includes fold and profile indices
        plot_filename = os.path.join(output_folder, f"Fold_{fold_index+1}_Profile_{profile_index+1}.png")
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to save memory
    else:
        print(f"Dimension mismatch in plot_profile: X_valid={len(X_valid)}, Y_true_valid={len(Y_true_valid)}, "
              f"baseline_pred={len(baseline_pred)}, Y_gpr_pred={len(Y_gpr_pred)}, Y_gpr_std={len(Y_gpr_std)}")


# Directory where CSV files are located
file_directory = r'B:\dataset_slicing\optimized_files\optimized_files'  # Replace with your actual path

# Set lower and upper bounds for acceptable height values
LOWER_THRESHOLD = 200
UPPER_THRESHOLD = 520

# Load data
all_measurements, min_profiles, file_paths = load_data(file_directory)

# Set up KFold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Initialize dictionary to store results across profiles and folds
profile_results = {}

# Function to process each profile with GPR
def process_profile_gpr(profile_index, train_data, val_data, fold_index, file_directory):
    # Initialisiere eine Liste zur Speicherung der Ergebnisse pro Validierungsdatei
    results_per_val_file = []

    # Berechnung des Mittelwertprofils über die Trainingsdateien
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

    # Wenn alle Trainingsdaten nach dem Filtern NaN sind, Profil überspringen
    if np.isnan(profile_data_train).all():
        return results_per_val_file  # Leere Liste zurückgeben, wenn keine Daten vorhanden sind

    # Mittelwertprofil über die Trainingsdateien berechnen
    baseline_profile = np.nanmean(profile_data_train, axis=0)
    Y_train = baseline_profile.copy()
    X = np.arange(Y_train.shape[0])

    # Entferne NaN-Werte aus den Trainingsdaten
    valid_train_idx = ~np.isnan(Y_train)
    X_valid = X[valid_train_idx].reshape(-1, 1)
    Y_train_valid = Y_train[valid_train_idx].reshape(-1, 1)

    if len(Y_train_valid) < 2:
        return results_per_val_file  # Leere Liste zurückgeben, wenn unzureichende Daten vorliegen

    # Skalierung von X und Y_train
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_valid)
    Y_scaled = scaler_Y.fit_transform(Y_train_valid)

    # Definition und Training des GPR-Modells
    kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=3)
    model = GPy.models.GPRegression(X_scaled, Y_scaled, kernel)
    model.optimize(messages=False)

    # Erzeuge Vorhersagearray und berechne Standardabweichungen
    full_Y_pred = np.full(len(Y_train), np.nan)
    full_Y_std = np.full(len(Y_train), np.nan)
    Y_pred_scaled, Y_var_scaled = model.predict(X_scaled)
    Y_std_scaled = np.sqrt(Y_var_scaled)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    Y_std = scaler_Y.scale_ * Y_std_scaled.flatten()
    full_Y_pred[valid_train_idx] = Y_pred.flatten()
    full_Y_std[valid_train_idx] = Y_std

    # Berechne GPR-Vorhersagen und Metriken für jede Datei in den Validierungsdaten

    # In der Schleife für jede Validierungsdatei in process_profile_gpr
    for val_file_idx, val_file in enumerate(val_data):
        val_file_name = file_paths[val_file_idx]  # Dateiname der aktuellen Validierungsdatei

        val_profile = np.where(
            (val_file[profile_index, :] >= LOWER_THRESHOLD) & 
            (val_file[profile_index, :] <= UPPER_THRESHOLD),
            val_file[profile_index, :],
            np.nan
        )

        valid_idx = ~np.isnan(val_profile) & ~np.isnan(full_Y_pred)
        if valid_idx.sum() < 2:
            continue  # Überspringen, wenn unzureichende gültige Daten vorhanden sind

        val_profile_valid = val_profile[valid_idx]
        pred_valid = full_Y_pred[valid_idx]
        baseline_valid = Y_train[valid_idx]
        std_valid = full_Y_std[valid_idx]

        # Plot nur für einen bestimmten Profilbereich erstellen, z.B. zwischen 90 und 100
        if PLOT_PROFILES and (90 <= profile_index < 100):
            plot_profile(X, val_profile, baseline_valid, pred_valid, std_valid, 
                        profile_index, fold_index, val_file_idx, file_directory, val_file_name)

        # Berechne die Metriken für diese Validierungsdatei
        rmse_gpr, mae_gpr, r2_gpr = calculate_metrics(val_profile_valid, pred_valid)
        rmse_baseline, mae_baseline, r2_baseline = calculate_metrics(val_profile_valid, baseline_valid)

        # Berechne den p-Wert basierend auf den Residuen
        residuals_gpr = val_profile_valid - pred_valid
        residuals_baseline = val_profile_valid - baseline_valid
        p_value = None
        if len(residuals_gpr) > 1 and len(residuals_baseline) > 1:
            t_stat, p_value = ttest_rel(residuals_gpr, residuals_baseline)

        # Speichere die Ergebnisse für diese Validierungsdatei
        results_per_val_file.append({
            "Validation File": val_file_idx + 1,
            "RMSE GPR": rmse_gpr,
            "MAE GPR": mae_gpr,
            "R2 GPR": r2_gpr,
            "RMSE Baseline": rmse_baseline,
            "MAE Baseline": mae_baseline,
            "R2 Baseline": r2_baseline,
            "P-value": p_value
        })

    return results_per_val_file

# Iterate over each fold
for fold_index, (train_indices, val_indices) in enumerate(kf.split(all_measurements)):
    print(f"\nProcessing Fold {fold_index+1}/{kf.get_n_splits()}")
    train_data = [all_measurements[i] for i in train_indices]
    val_data = [all_measurements[i] for i in val_indices]

    # Run GPR profile processing in parallel for this fold
    results_gpr = Parallel(n_jobs=-1)(
        delayed(process_profile_gpr)(i, train_data, val_data, fold_index, file_directory) 
        for i in tqdm(range(min_profiles), desc=f"Processing Profiles for Fold {fold_index+1}")
    )

    # Collect results for each validation file and store them
    for profile_index, profile_results_per_val in enumerate(results_gpr):
        for val_file_result in profile_results_per_val:
            profile_key = f"Fold_{fold_index + 1}_Validation_File_{val_file_result['Validation File']}_Profile_{profile_index + 1}"
            profile_results[profile_key] = val_file_result

# Save the results to a JSON file
output_file = os.path.join(file_directory, "profile_metrics.json")
with open(output_file, "w") as f:
    json.dump(profile_results, f, indent=4)

print(f"\nResults for each profile and validation file have been saved to {output_file}")