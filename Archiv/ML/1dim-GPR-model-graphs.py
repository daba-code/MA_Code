import pandas as pd
import numpy as np
import GPy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import glob

# Funktion zur Berechnung des RMSE
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Funktion zur Berechnung des R^2 Werts
def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

# Funktion zur Berechnung des MAE
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Verzeichnis, in dem sich die CSV-Dateien befinden
file_directory = r'B:\dataset_slicing\optimized_files'  # Ersetze dies mit dem tatsächlichen Pfad zu deinen Dateien

# Schritt 1: Laden aller CSV-Dateien und Bestimmung der minimalen Zeilenanzahl
file_paths = glob.glob(f"{file_directory}/*.csv")
all_measurements = []
min_profiles = float('inf')  # Startwert für die minimale Anzahl an Profilen

# Einlesen der Daten und Bestimmen der minimalen Zeilenanzahl
for file_path in file_paths:
    df = pd.read_csv(file_path, sep=";", header=None)
    min_profiles = min(min_profiles, df.shape[0])  # Aktualisieren der minimalen Anzahl an Profilen
    all_measurements.append(df)

# Konvertiere min_profiles zu einer Ganzzahl
min_profiles = int(min_profiles)

# Kürzen aller Dateien auf die minimale Zeilenanzahl
all_measurements = [df.iloc[:min_profiles, :].values for df in all_measurements]

# Aufteilung der Daten in Training (80%) und Validierung (20%)
train_size = int(0.8 * len(all_measurements))
train_data = all_measurements[:train_size]    # Training: 80% der Dateien
val_data = all_measurements[train_size:]      # Validierung: 20% der Dateien

# Initialisierung der Zählung für verwendete Zeilen
used_rows = set()

# Initialisierung des Plots
plt.ion()  # Interactive mode for dynamic plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Set lower and upper bounds for acceptable height values
LOWER_THRESHOLD = 200
UPPER_THRESHOLD = 520

# Iterate through each profile, applying the thresholds
for profile_index in range(min_profiles):
    profile_data_train = []

    for train_file in train_data:
        # Apply both lower and upper bounds: values outside these bounds are set to NaN
        filtered_profile = np.where(
            (train_file[profile_index, :] >= LOWER_THRESHOLD) & 
            (train_file[profile_index, :] <= UPPER_THRESHOLD),
            train_file[profile_index, :],
            np.nan
        )
        
        profile_data_train.append(filtered_profile)

    # Convert the filtered profiles list to an array if needed for further processing
    profile_data_train = np.array(profile_data_train)
    # Check if all entries for the current profile row are NaN across all training files
    if all(np.isnan(profile).all() for profile in profile_data_train):
        print(f"Profil {profile_index + 1} wurde vom Training und der Validierung ausgeschlossen - keine gültigen Daten.")
        continue

    print(f"\nTrainiere GPR-Modell für Profil {profile_index + 1} über alle Trainingsdateien")
    used_rows.add(profile_index)

    # Berechnung von X und Y_train unter Beachtung der NaN-Werte
    num_columns = profile_data_train[0].shape[0]
    X = np.arange(num_columns).reshape(-1, 1)  # Position entlang des Profils (0 bis Anzahl Spalten)
    Y_train = np.nanmean(profile_data_train, axis=0).reshape(-1, 1)  # Durchschnittswerte über die Dateien für das Profil

    # Skip profiles where Y_train is entirely NaN after processing
    if np.isnan(Y_train).all():
        print(f"Profil {profile_index + 1} wurde ausgeschlossen - keine gültigen Daten nach der Mittelwertberechnung.")
        continue

    # Definition eines GPR-Modells für das Profil mit RBF-Kernel
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=20.)
    model = GPy.models.GPRegression(X, Y_train, kernel)
    model.optimize(messages=False)

    # Validierung auf dem Profil in den Validierungsdaten und Berechnung von RMSE, R^2 und MAE
    rmse_scores, r2_scores, mae_scores = [], [], []
    for val_file in val_data:
        # Ersetzen von Werten < 200 durch NaN in den Validierungsprofilen
        val_profile = np.where(val_file[profile_index, :] >= 200, val_file[profile_index, :], np.nan).reshape(-1, 1)

        # Filtern von gültigen Indizes, um Berechnung von Metriken auf nur gültigen Daten durchzuführen
        valid_idx = ~np.isnan(val_profile.flatten())
        
        # Skip this validation profile if no valid data points are found
        if not valid_idx.any():
            print(f"Validierungsprofil für Profil {profile_index + 1} übersprungen - keine gültigen Werte.")
            continue

        Y_pred, _ = model.predict(X)
        
        # Ensure there are no NaN values in Y_pred for valid indices before calculating metrics
        if np.isnan(Y_pred[valid_idx]).any():
            print(f"Profil {profile_index + 1} enthält NaN-Werte in den Vorhersagen, überspringe Metriken.")
            continue
        
        rmse = root_mean_squared_error(val_profile[valid_idx], Y_pred[valid_idx])
        r2_value = r2(val_profile[valid_idx], Y_pred[valid_idx])
        mae_value = mae(val_profile[valid_idx], Y_pred[valid_idx])
        
        rmse_scores.append(rmse)
        r2_scores.append(r2_value)
        mae_scores.append(mae_value)
    
    # Berechnung des durchschnittlichen RMSE, R^2 und MAE für das Profil über die Validierungsdateien
    if rmse_scores:
        avg_rmse = np.mean(rmse_scores)
        avg_r2 = np.mean(r2_scores)
        avg_mae = np.mean(mae_scores)
        print(f"Profil {profile_index + 1} - Durchschnittlicher RMSE: {avg_rmse:.4f}")
        print(f"Profil {profile_index + 1} - Durchschnittlicher R^2: {avg_r2:.4f}")
        print(f"Profil {profile_index + 1} - Durchschnittlicher MAE: {avg_mae:.4f}")

    # Aktualisierung des Plots
    ax.clear()
    ax.plot(X, Y_train, color='red', label='GPR Mittelwert')
    ax.fill_between(
        X.flatten(),
        (Y_train - 1.96 * np.nanstd(Y_train)).flatten(),
        (Y_train + 1.96 * np.nanstd(Y_train)).flatten(),
        color='orange',
        alpha=0.2,
        label='95% Konfidenzintervall'
    )

    # Datenpunkte für das Profil in den Trainingsdateien
    for i, train_data_profile in enumerate(profile_data_train):
        ax.scatter(X, train_data_profile, label=f'Trainingsdatei {i + 1}', alpha=0.6, s=10)

    ax.set_title(f"GPR Modell für Profil {profile_index + 1} (80% Training, 20% Validierung)")
    ax.set_xlabel("Position entlang des Profils")
    ax.set_ylabel("Höhenwert")
    ax.legend()

    plt.pause(1)  # Pause to display each profile plot dynamically

# Ausgabe der Gesamtanzahl verwendeter Zeilen
print(f"\nGesamtanzahl der für das Training verwendeten Profile: {len(used_rows)}")

# Ensure the final plot persists
plt.ioff()
plt.show()
