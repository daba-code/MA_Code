import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.signal import savgol_filter

def load_csv_files(directory):
    file_paths = glob(os.path.join(directory, "*.csv"))
    data_frames = []
    
    for file_path in file_paths:
        df = pd.read_csv(file_path, delimiter=';', header=None, skip_blank_lines=True)
        data_frames.append(df)
        print(f"Datei geladen: {file_path} mit {df.shape[0]} Zeilen und {df.shape[1]} Spalten.")
    
    return data_frames

def calculate_median_profile(data_frames, window, min_valid_value):
    stacked = np.stack([df.values for df in data_frames], axis=2)
    stacked[stacked < min_valid_value] = np.nan
    median_profiles = np.nanmedian(stacked, axis=2)

    median_profile_df = pd.DataFrame(median_profiles)
    
    # Rollen für glatte Übergänge und sicherstellen, dass Spalten ohne gültige Werte interpoliert werden
    rolling_median_profile = median_profile_df.T.rolling(window=window, min_periods=1).median().T
    rolling_median_profile = rolling_median_profile.interpolate(method='linear', axis=1, limit_direction='both')
    
    return rolling_median_profile

def calculate_thresholds(median_profile, threshold_factor):
    std_deviation = median_profile.std(axis=1).to_numpy()
    upper_threshold = median_profile.to_numpy() + threshold_factor * std_deviation[:, None]
    lower_threshold = median_profile.to_numpy() - threshold_factor * std_deviation[:, None]
    
    return pd.DataFrame(upper_threshold), pd.DataFrame(lower_threshold)

def apply_thresholds(data_frames, upper_threshold, lower_threshold):
    for df in data_frames:
        mask = (df > upper_threshold) | (df < lower_threshold)
        df[mask] = np.nan

    return data_frames

def process_row(row, median_row):
    """
    Verarbeitet eine einzelne Zeile und füllt NaNs anhand des Medianprofils und
    des Median-Abstands vorhandener Werte auf.
    """
    row_filled = row.copy()
    valid_values = row[~np.isnan(row)]
    if len(valid_values) == 0:
        return row  # Keine gültigen Werte, keine Füllung möglich

    # Berechnung des Median-Abstands
    median_deviation = np.median(valid_values - median_row[~np.isnan(row)])
    row_filled = row_filled.fillna(median_row + median_deviation)
    return row_filled

def process_row(row, median_row, weight_factor=0.7):
    """
    Füllt NaN-Werte in einer einzelnen Zeile auf, basierend auf dem Medianabstand der vorhandenen Werte.
    """
    # Gültige Werte extrahieren
    valid_values = row[~row.isna()]
    median_values = median_row[~row.isna()]

    # Berechne den Durchschnitt der Differenz zwischen den vorhandenen Werten und dem Medianprofil
    if len(valid_values) > 0:
        median_diff = (valid_values - median_values).mean()

        # NaN-Werte füllen und dabei die Gewichtung anwenden
        row_filled = row.fillna(median_row + weight_factor * median_diff)
    else:
        row_filled = row.fillna(median_row)  # Fallback, falls keine gültigen Werte vorhanden sind
    
    return row_filled

def fill_with_median_and_interpolation(data_frames, median_profile):
    filled_data_frames = []
    
    with tqdm(total=len(data_frames), desc="Füllen der NaN-Werte", unit="Datei") as pbar:
        for df in data_frames:
            filled_df = df.copy()
            
            # Paralleles Rechnen für jede Zeile
            filled_rows = Parallel(n_jobs=-1)(delayed(process_row)(filled_df.iloc[row_idx], median_profile.iloc[row_idx])
                                              for row_idx in range(filled_df.shape[0]))
            
            # Zusammenfügen der verarbeiteten Zeilen
            filled_df = pd.DataFrame(filled_rows, columns=filled_df.columns)
            
            # Lineare Interpolation als letzte Füllung
            filled_df = filled_df.interpolate(method='linear', axis=1, limit_direction='both')
            
            # Moving Average zur Glättung (kann auf größere Fenster eingestellt werden)
            filled_df = filled_df.rolling(window=10, min_periods=1, axis=1).mean()
            
            # Savitzky-Golay-Filter für finale Glättung
            for row_idx in range(filled_df.shape[0]):
                filled_df.iloc[row_idx] = savgol_filter(filled_df.iloc[row_idx], window_length=7, polyorder=2)
                
            filled_data_frames.append(filled_df)
            pbar.update(1)
    
    return filled_data_frames


def plot_processed_profiles(filled_data_frames, median_profile, upper_threshold, lower_threshold):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))

    for profile_idx in range(len(filled_data_frames[0])):
        ax.clear()
        
        for file_idx, df in enumerate(filled_data_frames):
            ax.plot(df.iloc[profile_idx], label=f"Datei {file_idx + 1}")

        ax.plot(median_profile.iloc[profile_idx], 'k--', label="Median-Profil")
        ax.plot(upper_threshold.iloc[profile_idx], 'r--', label="Oberer Schwellenwert")
        ax.plot(lower_threshold.iloc[profile_idx], 'b--', label="Unterer Schwellenwert")

        ax.set_title(f"Bearbeitete Daten - Profil {profile_idx + 1}")
        ax.set_xlabel("Position entlang des Profils")
        ax.set_ylabel("Wert")
        ax.legend()
        
        plt.draw()
        plt.pause(2)

    plt.ioff()
    plt.show()

def check_for_nan(data_frames):
    for file_idx, df in enumerate(data_frames):
        if df.isna().any().any():
            print(f"NaN-Werte in Datei {file_idx + 1} gefunden")
        else:
            print(f"Datei {file_idx + 1} hat keine NaN-Werte")

def process_files(directory, plot_processed=True, window=25, min_valid_value=300, threshold_factor=2):
    data_frames = load_csv_files(directory)
    
    if not data_frames:
        print(f"Keine CSV-Dateien im Verzeichnis '{directory}' gefunden.")
        return
    
    median_profile = calculate_median_profile(data_frames, window, min_valid_value)
    upper_threshold, lower_threshold = calculate_thresholds(median_profile, threshold_factor)
    data_frames = apply_thresholds(data_frames, upper_threshold, lower_threshold)
    
    filled_data_frames = fill_with_median_and_interpolation(data_frames, median_profile)
    check_for_nan(filled_data_frames)
    
    if plot_processed:
        plot_processed_profiles(filled_data_frames, median_profile, upper_threshold, lower_threshold)

# Beispielaufruf
directory = r'B:\filtered_output\gekuerzt\aligned_files'
process_files(directory)
