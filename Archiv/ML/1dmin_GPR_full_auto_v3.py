import pandas as pd
import numpy as np
import GPy
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from tqdm import tqdm
import glob
import warnings
import json
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Plotting flag
PLOT_PROFILES = True  # Set to True to enable plotting

# Functions for calculating metrics
def calculate_metrics(y_true, y_pred):
    valid_idx = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_valid = y_true[valid_idx]
    y_pred_valid = y_pred[valid_idx]
    
    if len(y_true_valid) > 1 and len(y_pred_valid) > 1:
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        r2 = r2_score(y_true_valid, y_pred_valid)
    else:
        rmse, mae, r2 = np.nan, np.nan, np.nan

    return rmse, mae, r2

def plot_specific_profile_with_gpr(all_measurements, profile_results, file_directory, profile_index):
    """
    Visualisiert die Trainingsdatenpunkte, GPR-Mittelwert und das 95%-Konfidenzintervall für ein spezifisches Profil.

    :param all_measurements: Liste der Trainingsdatensätze
    :param profile_results: Dictionary mit den GPR-Vorhersagen und Konfidenzintervallen pro Fold und Profil
    :param file_directory: Verzeichnis, in dem die Plots gespeichert werden sollen
    :param profile_index: Der Index des Profils, das geplottet werden soll
    """
    output_folder = os.path.join(file_directory, "specific_profile_gpr_plot")
    os.makedirs(output_folder, exist_ok=True)

    for key, metrics in profile_results.items():
        fold_info, val_file, profile = key.split("_")[:3]
        
        if "Profile" in key:
            profile = key.split("Profile")[1].strip()
            # Entferne alle nicht-numerischen Zeichen und konvertiere in eine Ganzzahl
            profile_number = int(''.join(filter(str.isdigit, profile)))
            if profile_number != profile_index:
                continue  # Überspringe Profile, die nicht dem angegebenen Index entsprechen
        else:
            continue  # Überspringe Einträge ohne "Profile" im key

        title = f"{fold_info} - {val_file} - Profile {profile_index + 1}"

        # Extrahiere Trainingsdatenpunkte für das spezifische Profil
        training_data_points = [measurement.iloc[profile_index, :].values for measurement in all_measurements]
        
        # Konvertiere zu einem Array und wende Schwellenwerte an
        training_data_points = np.array(training_data_points)
        training_data_points = np.where(
            (training_data_points >= LOWER_THRESHOLD) & 
            (training_data_points <= UPPER_THRESHOLD), 
            training_data_points, np.nan
        )

        # GPR-Vorhersagen und Konfidenzintervall extrahieren
        X_valid = np.arange(training_data_points.shape[1])  # Annahme: X ist die Position entlang des Profils
        gpr_mean = np.array(metrics["GPR Mean"])
        gpr_std = np.array(metrics["GPR Std"])


        # Plot der Trainingsdatenpunkte und GPR-Ergebnisse
        plt.figure(figsize=(10, 6))
        plt.plot(X_valid, np.nanmean(training_data_points, axis=0), label="Trainingsdaten (Mittelwert)", color="blue", linestyle="--", alpha=0.7)
        for train_data in training_data_points:
            plt.scatter(X_valid, train_data, color="blue", alpha=0.1)  # Plot der einzelnen Datenpunkte

        plt.plot(X_valid, gpr_mean, label="GPR Mean Prediction", color="red")
        plt.fill_between(
            X_valid,
            gpr_mean - 1.96 * gpr_std,
            gpr_mean + 1.96 * gpr_std,
            color="orange", alpha=0.5, label="95% Confidence Interval"
        )

        plt.title(f"{title}: Training Data Points and GPR Prediction")
        plt.xlabel("Position entlang des Profils")
        plt.ylabel("Höhe")
        plt.legend()

        plot_filename = os.path.join(output_folder, f"{title}_TrainingData_GPR.png")
        plt.savefig(plot_filename)
        plt.close()

        print(f"Plot saved for {title}")
        break  # Nur ein Plot für das angegebene Profil erstellen

def load_data(file_directory):
    # Lade alle CSV-Dateien
    file_paths = glob.glob(f"{file_directory}/*.csv")
    all_measurements = []

    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=";", header=None)
        all_measurements.append(df)

    # Anwenden der Schwellenwerte und Filtern nach NaN-Konsistenz
    all_measurements = apply_thresholds_and_filter(all_measurements)

    return all_measurements, file_paths

def apply_thresholds_and_filter(all_measurements, lower_threshold=200, upper_threshold=520):
    # Konvertiere zu einem 3D-Array (Anzahl Dateien, Reihen, Spalten)
    data_stack = np.array([df.values for df in all_measurements])

    # Schwellenwerte anwenden
    data_stack = np.where((data_stack >= lower_threshold) & (data_stack <= upper_threshold), data_stack, np.nan)

    # Durchschnittswerte für NaN-Werte berechnen
    for i in range(data_stack.shape[0]):  # für jede Datei
        for j in range(data_stack.shape[1]):  # für jede Reihe
            for k in range(data_stack.shape[2]):  # für jede Spalte
                if np.isnan(data_stack[i, j, k]):
                    left_index = max(0, k - 20)
                    right_index = min(data_stack.shape[2], k + 20)
                    surrounding_values = data_stack[i, j, left_index:k].tolist() + data_stack[i, j, k + 1:right_index].tolist()

                    # Suche auch in den benachbarten Reihen
                    for offset in [-1, 0, 1]:  # -1 = vorherige Reihe, 0 = aktuelle Reihe, 1 = nachfolgende Reihe
                        if 0 <= j + offset < data_stack.shape[1]:  # Überprüfen, ob die Reihe im gültigen Bereich ist
                            surrounding_values += data_stack[i, j + offset, left_index:right_index].tolist()

                    # NaN-Werte herausfiltern
                    surrounding_values = [v for v in surrounding_values if not np.isnan(v)]

                    if surrounding_values:
                        data_stack[i, j, k] = np.mean(surrounding_values)
                    else:
                        # Suche in der nächsten Reihe, falls immer noch NaN
                        if j + 1 < data_stack.shape[1]:
                            next_row_values = data_stack[i, j + 1, k]
                            if not np.isnan(next_row_values):
                                data_stack[i, j, k] = next_row_values

    # Rückkonvertierung zu Liste von DataFrames
    return [pd.DataFrame(data_stack[i]) for i in range(data_stack.shape[0])]

def plot_profile(X, Y_true, baseline_pred, Y_gpr_pred, Y_gpr_std, profile_index, fold_index, val_file_name, file_directory):
    valid_idx = ~np.isnan(Y_true) & ~np.isnan(baseline_pred) & ~np.isnan(Y_gpr_pred) & ~np.isnan(Y_gpr_std)

    if not np.any(valid_idx):
        print(f"Skipping plot for Fold {fold_index + 1}, Profile {profile_index + 1} - No valid data.")
        return  

    X_valid = X[valid_idx]
    Y_true_valid = Y_true[valid_idx]
    baseline_pred_valid = baseline_pred[valid_idx]
    Y_gpr_pred_valid = Y_gpr_pred[valid_idx]
    Y_gpr_std_valid = Y_gpr_std[valid_idx]

    lengths = {len(arr) for arr in [X_valid, Y_true_valid, baseline_pred_valid, Y_gpr_pred_valid, Y_gpr_std_valid]}
    if len(lengths) != 1:
        raise ValueError(f"Arrays have different lengths in plot for Fold {fold_index + 1}, Profile {profile_index + 1}!")

    plt.figure(figsize=(10, 6))
    plt.plot(X_valid, Y_true_valid, label="Actual Values", color="blue")
    plt.plot(X_valid, baseline_pred_valid, label="Baseline Prediction", linestyle="--", color="green")
    plt.plot(X_valid, Y_gpr_pred_valid, label="GPR Prediction", color="red")
    plt.fill_between(
        X_valid.flatten(),
        (Y_gpr_pred_valid.flatten() - 1.96 * Y_gpr_std_valid),
        (Y_gpr_pred_valid.flatten() + 1.96 * Y_gpr_std_valid),
        color="orange", alpha=0.2, label="95% Confidence Interval (GPR)"
    )

    clean_val_file_name = os.path.splitext(os.path.basename(val_file_name))[0]
    plt.title(f"Fold {fold_index + 1} - {clean_val_file_name} - Profile {profile_index + 1}: Baseline vs GPR Predictions")
    plt.xlabel("Position entlang des Profils")
    plt.ylabel("Höhe")
    plt.legend()

    output_folder = os.path.join(file_directory, "plots")
    os.makedirs(output_folder, exist_ok=True)

    plot_filename = os.path.join(output_folder, f"Fold_{fold_index + 1}_Profile_{profile_index + 1}.png")
    plt.savefig(plot_filename)
    plt.close()

# Directory where CSV files are located
file_directory = r'B:\filtered_output\gekuerzt\aligned_files\processed_files\gekuerzt'  # Replace with your actual path

LOWER_THRESHOLD = 200
UPPER_THRESHOLD = 520

all_measurements, file_paths = load_data(file_directory)
kf = KFold(n_splits=3, shuffle=True, random_state=42)
profile_results = {}

def process_profile_gpr(profile_index, train_data, val_data, fold_index, file_directory):
    results_per_val_file = []

    profile_data_train = []
    for train_file in train_data:
        profile = np.where(
            (train_file.iloc[profile_index, :] >= LOWER_THRESHOLD) & 
            (train_file.iloc[profile_index, :] <= UPPER_THRESHOLD),
            train_file.iloc[profile_index, :],
            np.nan
        )
        profile_data_train.append(profile)

    profile_data_train = np.array(profile_data_train)
    baseline_profile = np.nanmean(profile_data_train, axis=0)
    Y_train = baseline_profile.copy()
    X = np.arange(Y_train.shape[0])

    valid_train_idx = ~np.isnan(Y_train)
    X_valid = X[valid_train_idx].reshape(-1, 1)
    Y_train_valid = Y_train[valid_train_idx].reshape(-1, 1)

    if np.any(np.isnan(X_valid)) or np.any(np.isnan(Y_train_valid)):
        print("Found NaN values in the training data!")
    if np.any(np.isinf(X_valid)) or np.any(np.isinf(Y_train_valid)):
        print("Found Inf values in the training data!")

    if len(Y_train_valid) < 2:
        return results_per_val_file

    kernel = GPy.kern.RBF(input_dim=1, variance=20, lengthscale=50)
    model = GPy.models.GPRegression(X_valid, Y_train_valid, kernel)
    model.likelihood.variance = 1  # Fügt Jitter hinzu
    model.optimize(messages=False)

    full_Y_pred = np.full(len(Y_train), np.nan)
    full_Y_std = np.full(len(Y_train), np.nan)
    Y_pred, Y_var = model.predict(X_valid)
    Y_std = np.sqrt(Y_var).flatten()

    full_Y_pred[valid_train_idx] = Y_pred.flatten()
    full_Y_std[valid_train_idx] = Y_std

    for val_file_idx, val_file in enumerate(val_data):
        val_file_name = file_paths[val_indices[val_file_idx]]  # Korrektes Mapping des Dateinamens

        val_profile = np.where(
            (val_file.iloc[profile_index, :] >= LOWER_THRESHOLD) & 
            (val_file.iloc[profile_index, :] <= UPPER_THRESHOLD),
            val_file.iloc[profile_index, :],
            np.nan
        )

        valid_idx = ~np.isnan(val_profile) & ~np.isnan(full_Y_pred) & ~np.isnan(baseline_profile)

        val_profile_valid = val_profile[valid_idx]
        pred_valid = full_Y_pred[valid_idx]
        baseline_valid = baseline_profile[valid_idx]
        std_valid = full_Y_std[valid_idx]

        if PLOT_PROFILES and (90 <= profile_index < 100):
            plot_profile(X, val_profile, baseline_valid, pred_valid, std_valid, 
                         profile_index, fold_index, val_file_name, file_directory)

        rmse_gpr, mae_gpr, r2_gpr = calculate_metrics(val_profile_valid, pred_valid)
        rmse_baseline, mae_baseline, r2_baseline = calculate_metrics(val_profile_valid, baseline_valid)

        residuals_gpr = val_profile_valid - pred_valid
        residuals_baseline = val_profile_valid - baseline_valid
        p_value = None
        if len(residuals_gpr) > 1 and len(residuals_baseline) > 1:
            t_stat, p_value = ttest_rel(residuals_gpr, residuals_baseline)

        results_per_val_file.append({
            "Validation File": val_file_name,
            "Mean GPR Prediction": np.mean(full_Y_pred),  # Mittelwert der GPR-Vorhersage
            "Mean GPR Uncertainty": np.mean(full_Y_std),  # Mittelwert der GPR-Unsicherheit
            "RMSE GPR": rmse_gpr,
            "MAE GPR": mae_gpr,
            "R2 GPR": r2_gpr,
            "RMSE Baseline": rmse_baseline,
            "MAE Baseline": mae_baseline,
            "R2 Baseline": r2_baseline,
            "P-value": p_value
        })

    return results_per_val_file

for fold_index, (train_indices, val_indices) in enumerate(kf.split(all_measurements)):
    print(f"\nProcessing Fold {fold_index + 1}/{kf.get_n_splits()}")
    
    train_data = [all_measurements[i] for i in train_indices]
    val_data = [all_measurements[i] for i in val_indices]

    print(f"Train indices for Fold {fold_index + 1}: {train_indices}")
    print(f"Validation indices for Fold {fold_index + 1}: {val_indices}")
    print(f"Validation files for Fold {fold_index + 1}: {[file_paths[i] for i in val_indices]}")

    results_gpr = Parallel(n_jobs=-1)(
        delayed(process_profile_gpr)(i, train_data, val_data, fold_index, file_directory) 
        for i in tqdm(range(len(all_measurements[0])), desc=f"Processing Profiles for Fold {fold_index + 1}")
    )

    for profile_index, profile_results_per_val in enumerate(results_gpr):
        if profile_results_per_val:
            for val_file_result in profile_results_per_val:
                val_file_name = os.path.basename(val_file_result["Validation File"])
                profile_key = f"Fold_{fold_index + 1}_Validation_File_{val_file_name}_Profile_{profile_index + 1}"
                profile_results[profile_key] = val_file_result

specific_profile_index = 150  # Setze das gewünschte Profil hier (Index basiert auf 0)
#plot_specific_profile_with_gpr(all_measurements, profile_results, file_directory, specific_profile_index)

output_file = os.path.join(file_directory, "profile_metrics.json")
with open(output_file, "w") as f:
    json.dump(profile_results, f, indent=4)

print(f"\nResults for each profile and validation file have been saved to {output_file}")
