import pandas as pd
import numpy as np
import glob
import GPy
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

# Funktion zum Laden der Daten in der gewünschten Reihe aus allen Dateien
def load_file(file_path):
    return pd.read_csv(file_path, sep=";", header=None)

# Funktion zum Anwenden des GPR und Rückgabe des geglätteten Profils
def gpr_smooth_row(x_positions, row_data, model):
    valid_indices = ~np.isnan(row_data)
    if not valid_indices.any() or np.all(row_data[valid_indices] == row_data[valid_indices][0]):
        return None

    X_train = x_positions[valid_indices].reshape(-1, 1)
    y_train = row_data[valid_indices].reshape(-1, 1)

    model.set_XY(X_train, y_train)
    y_pred, _ = model.predict(x_positions.reshape(-1, 1))
    return np.round(y_pred.flatten()).astype(int)

# Funktion zur Berechnung des y-Shift
def apply_y_shift(gpr_profile):
    middle_value = np.nanmean(gpr_profile[300:351])
    if np.isnan(middle_value):
        return None
    return gpr_profile - middle_value + 200

# Funktion zur Bestimmung des Nahtbereichs über ein Fenster
def find_weld_seam_area(profile, window_size=100):
    min_value = np.inf
    min_index = None
    for start in range(len(profile) - window_size + 1):
        window = profile[start:start + window_size]
        mean_value = np.nanmean(window)
        if mean_value < min_value:
            min_value = mean_value
            min_index = start 
    return min_index

# Funktion zur Berechnung des Mittelwertprofils und des allgemeinen Nahtbeginns
def calculate_reference_weld_position(gpr_profiles, window_size=100):
    mean_profile = np.nanmean(gpr_profiles, axis=0)
    return mean_profile, find_weld_seam_area(mean_profile, window_size)

# Funktion zum Plotten des Mittelwertprofils mit Nahtbereich
def plot_mean_profile_with_seam(mean_profile, seam_position, window_size=100):
    plt.figure(figsize=(12, 6))
    plt.plot(mean_profile, label='Mittelwertprofil')
    plt.axvspan(seam_position - window_size // 2, seam_position + window_size // 2, 
                color='orange', alpha=0.3, label='Erkannter Nahtbereich')
    plt.xlabel('x-Wert (Position)')
    plt.ylabel('Mittelwert Höhe')
    plt.title('Mittelwertprofil mit Nahtbereich')
    plt.legend()
    plt.show()

# Funktion zur Anwendung des x-Shift basierend auf dem allgemeinen Nahtbereich und Speichern der Shifts
def apply_x_shift(adjusted_profile, seam_position, reference_position, row_index, shift_info, max_shift=40):
    drift_range = range(reference_position - max_shift, reference_position + max_shift + 1)
    x_shift = int(reference_position - seam_position if seam_position in drift_range else 0)
    shift_info[row_index] = {
        "shift_amount": x_shift,
        "target_column": int(reference_position)
    }
    return np.roll(adjusted_profile, x_shift)

# Funktion zur Verarbeitung und Speicherung des GPR-Predicted-Datensatzes mit Shifts
def process_file(file_path, output_directory, shift_info_path):
    df = load_file(file_path)
    x_positions = np.arange(df.shape[1])

    # Initiales GPR-Modell erstellen und optimieren
    kernel = GPy.kern.Matern32(input_dim=1, variance=1, lengthscale=1, ARD=True)
    kernel.variance.constrain_bounded(0.1, 1200)
    kernel.lengthscale.constrain_bounded(0.1, 40)
    initial_X_train = x_positions.reshape(-1, 1)
    initial_y_train = np.random.rand(len(x_positions)).reshape(-1, 1)
    model = GPy.models.GPRegression(initial_X_train, initial_y_train, kernel)
    model.optimize(messages=False)
    
    # Schritt 1: GPR-Predictions und Berechnung des allgemeinen Nahtbereichs
    gpr_smoothed_profiles = []
    for index, row in df.iterrows():
        if row.isna().sum() > len(row) / 3:
            continue
        gpr_profile = gpr_smooth_row(x_positions, row.values, model)
        if gpr_profile is not None:
            adjusted_profile = apply_y_shift(gpr_profile)
            gpr_smoothed_profiles.append(adjusted_profile)

    # Berechne das allgemeine Nahtbeginn im Mittelwertprofil und plotte
    gpr_smoothed_profiles = np.array(gpr_smoothed_profiles)
    mean_profile, reference_position = calculate_reference_weld_position(gpr_smoothed_profiles)
    plot_mean_profile_with_seam(mean_profile, reference_position)

    # Dictionary zur Speicherung der Verschiebungen pro Reihe
    shift_info = {}

    # Schritt 2: Wende den x-Shift basierend auf der Referenzposition an und speichere die Shift-Informationen
    aligned_profiles = []
    for row_index, adjusted_profile in enumerate(gpr_smoothed_profiles):
        if np.isnan(adjusted_profile).all():
            aligned_profiles.append(adjusted_profile)
        else:
            seam_pos = find_weld_seam_area(adjusted_profile)
            aligned_profile = apply_x_shift(adjusted_profile, seam_pos, reference_position, row_index, shift_info)
            aligned_profiles.append(aligned_profile)

    # Ergebnisse in einem DataFrame speichern
    aligned_df = pd.DataFrame(aligned_profiles)
    output_path = os.path.join(output_directory, os.path.basename(file_path).replace(".csv", "_gpr_aligned.csv"))
    aligned_df.to_csv(output_path, sep=";", index=False, header=False)
    print(f"GPR Predictions mit Shifts gespeichert: {output_path}")

    # Shift-Informationen in JSON-Datei speichern
    shift_info_path = os.path.join(output_directory, os.path.basename(file_path).replace(".csv", "_shift_info.json"))
    with open(shift_info_path, "w") as json_file:
        json.dump(shift_info, json_file, indent=4)
    print(f"Shift-Informationen gespeichert: {shift_info_path}")

# Hauptfunktion zur Verarbeitung aller Dateien im Verzeichnis
def process_all_files(file_directory):
    output_directory = os.path.join(file_directory, "gpr_aligned_predictions")
    os.makedirs(output_directory, exist_ok=True)

    file_paths = glob.glob(f"{file_directory}/*.csv")
    for file_path in file_paths:
        process_file(file_path, output_directory, shift_info_path=None)

# Beispielanwendung
file_directory = r"B:\filtered_output_NaN_TH"
process_all_files(file_directory)
