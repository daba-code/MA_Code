import pandas as pd
import numpy as np
import glob
import GPy
import os
import json
from tqdm import tqdm
from joblib import Parallel, delayed

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
    return np.round(y_pred.flatten()).astype(int)  # Konvertiert zu int

# Funktion zur Berechnung des y-Shift
def apply_y_shift(gpr_profile):
    middle_value = np.nanmean(gpr_profile[330:345])
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
def process_file(file_path, output_directory, shift_info_path=None):
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
        if row.isna().sum() > len(row) / 4:
            continue
        gpr_profile = gpr_smooth_row(x_positions, row.values, model)
        if gpr_profile is not None:
            gpr_smoothed_profiles.append(gpr_profile)

    # Berechne das allgemeine Nahtbeginn im Mittelwertprofil
    gpr_smoothed_profiles = np.array(gpr_smoothed_profiles)
    mean_profile, reference_position = calculate_reference_weld_position(gpr_smoothed_profiles)

    # Dictionary zur Speicherung der Verschiebungen pro Reihe
    shift_info = {}

    # Schritt 2: Wende den x-Shift basierend auf der Referenzposition an und speichere die Shift-Informationen
    x_aligned_profiles = []
    for row_index, adjusted_profile in enumerate(gpr_smoothed_profiles):
        if np.isnan(adjusted_profile).all():
            x_aligned_profiles.append(adjusted_profile)
        else:
            seam_pos = find_weld_seam_area(adjusted_profile)
            x_aligned_profile = apply_x_shift(adjusted_profile, seam_pos, reference_position, row_index, shift_info)
            x_aligned_profiles.append(x_aligned_profile)

    # Schritt 3: Wende den y-Shift an
    y_aligned_profiles = []
    for adjusted_profile in x_aligned_profiles:
        if np.isnan(adjusted_profile).all():
            y_aligned_profiles.append(adjusted_profile)
        else:
            y_aligned_profile = apply_y_shift(adjusted_profile)
            y_aligned_profiles.append(y_aligned_profile)

    # Entferne die ersten und letzten 10 Reihen, konvertiere zu int und speichere
    aligned_df = pd.DataFrame(y_aligned_profiles[40:-50]).astype(int)
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

    # Parallele Verarbeitung der Dateien mit joblib
    Parallel(n_jobs=-1)(delayed(process_file)(file_path, output_directory) for file_path in tqdm(file_paths))

# Beispielanwendung
file_directory = r"B:\filtered_output_NaN_TH"
process_all_files(file_directory)
