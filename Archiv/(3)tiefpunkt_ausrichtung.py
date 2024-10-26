import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import correlate
from joblib import Parallel, delayed

def load_csv_files(directory):
    """
    Lädt alle CSV-Dateien im Verzeichnis und speichert sie in einer Liste von DataFrames.
    """
    file_paths = glob(os.path.join(directory, "*.csv"))
    data_frames = []
    
    for file_path in file_paths:
        df = pd.read_csv(file_path, delimiter=';', header=None, skip_blank_lines=True)
        df = df.fillna(0)  # NaN-Werte durch 0 ersetzen
        data_frames.append(df)
        print(f"Datei geladen: {file_path} mit {df.shape[0]} Zeilen und {df.shape[1]} Spalten.")
    
    return data_frames

def calculate_median_profile(profiles):
    """
    Berechnet das Median-Profil über alle Profile, welches als Referenz für die Ausrichtung dient.
    """
    profiles = np.array(profiles)
    median_profile = np.median(profiles, axis=0)
    return median_profile

def find_u_shape_minimum_window(profile, window_size=70):
    """
    Identifiziert den Bereich um den Tiefpunkt der U-Form im Profil und berechnet das Medianfenster.
    """
    min_index = np.argmin(profile)
    start_index = max(0, min_index - window_size // 2)
    end_index = min(len(profile), min_index + window_size // 2)
    
    # Fensterwerte und deren Median berechnen
    window_values = profile[start_index:end_index]
    window_median = np.median(window_values)
    window_center_index = (start_index + end_index) // 2

    return window_center_index, window_median

def calculate_shift_x(profile, reference_min_index):
    """
    Berechnet die Verschiebung in x-Richtung basierend auf dem Index des Tiefpunkts der U-Form.
    """
    prof_min_index, _ = find_u_shape_minimum_window(profile)
    shift = reference_min_index - prof_min_index
    return shift

def apply_shift_x(profile, shift):
    """
    Wendet die berechnete Verschiebung in der x-Richtung an, füllt dabei entstehende Lücken mit 0.
    """
    if shift > 0:
        shifted_profile = np.pad(profile, (shift, 0), mode='constant', constant_values=0)[:len(profile)]
    elif shift < 0:
        shifted_profile = np.pad(profile, (0, -shift), mode='constant', constant_values=0)[-len(profile):]
    else:
        shifted_profile = profile  # Keine Verschiebung nötig
    return shifted_profile

def calculate_shift_y(profile, reference_min_value):
    """
    Berechnet die Verschiebung in y-Richtung basierend auf der Differenz zum Median im U-Fenster.
    Ignoriert dabei Nullwerte, die durch die x-Verschiebung entstanden sind.
    """
    # Fenster um den Tiefpunkt der U-Form finden und nur Nicht-Null-Werte im Fenster berücksichtigen
    _, prof_min_value = find_u_shape_minimum_window(profile)
    
    # Nur Nicht-Null-Werte für die Berechnung der y-Verschiebung verwenden
    non_zero_values = profile[profile != 0]
    if len(non_zero_values) > 0:
        prof_min_value = np.median(non_zero_values)
    else:
        prof_min_value = 0
    
    shift_y = reference_min_value - prof_min_value
    return shift_y

def apply_shift_y(profile, shift_y):
    """
    Wendet die berechnete Verschiebung in der y-Richtung an.
    """
    return profile + shift_y

def align_profiles_to_u_shape(profiles, reference_profile):
    """
    Richtet jedes Profil relativ zur U-Form des Referenzprofils in x- und y-Richtung aus.
    """
    aligned_profiles = []
    
    # Bestimme den Tiefpunkt-Bereich des Referenzprofils (Index und Median im Fenster)
    ref_min_index, ref_min_value = find_u_shape_minimum_window(reference_profile)
    
    for profile in profiles:
        # Berechne die x-Verschiebung und wende sie an
        shift_x = calculate_shift_x(profile, ref_min_index)
        aligned_profile = apply_shift_x(profile, shift_x)
        
        # Berechne die y-Verschiebung und wende sie an, wobei Nullwerte ignoriert werden
        shift_y = calculate_shift_y(aligned_profile, ref_min_value)
        aligned_profile = apply_shift_y(aligned_profile, shift_y)
        
        aligned_profiles.append(aligned_profile)
    
    return aligned_profiles

def process_profiles(data_frames):
    """
    Bereitet die x- und y-Richtung Korrektur für jedes Profil vor und führt diese aus.
    """
    max_profiles = min([df.shape[0] for df in data_frames])
    
    # Sammle alle Profile in einer Liste
    profiles = [
        [df.iloc[profile_idx].values for df in data_frames]
        for profile_idx in range(max_profiles)
    ]

    # Erstelle das Median-Profil als Referenz
    median_profile = calculate_median_profile([item for sublist in profiles for item in sublist])

    # Richte alle Profile an der U-Form des Median-Profils in x- und y-Richtung aus
    results = Parallel(n_jobs=-1)(
        delayed(align_profiles_to_u_shape)(profile_data, median_profile)
        for profile_data in profiles
    )
    
    return results

def plot_profiles(aligned_profiles, start_row=40):
    """
    Visualisiert die korrigierten Profile, beginnend ab der angegebenen Startreihe.
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))

    # Schleife startet bei der gewünschten Startreihe (start_row)
    for profile_idx, profile_data in enumerate(aligned_profiles[start_row:], start=start_row + 1):
        ax.clear()

        for file_idx, profile_values in enumerate(profile_data):
            ax.plot(profile_values, label=f"Datei {file_idx + 1}")

        ax.set_title(f"Profil {profile_idx} über alle Dateien hinweg")
        ax.set_xlabel("Position entlang des Profils")
        ax.set_ylabel("Wert")
        ax.legend()
        
        plt.draw()
        plt.pause(2)

    plt.ioff()
    plt.show()


# Beispielverwendung
directory = r'B:\filtered_output\gekuerzt'
data_frames = load_csv_files(directory)
aligned_profiles = process_profiles(data_frames)

# Visualisiere die korrigierten Profile
plot_profiles(aligned_profiles)
