from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, medfilt
import glob
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import os

def load_data(file_directory):
    file_paths = glob.glob(f"{file_directory}/*.csv")
    all_measurements = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=";", header=None)
        all_measurements.append((df, file_path))  # Speichern von DataFrame und Dateipfad als Tupel
    return all_measurements

def interpolate_nan_values(data, method='linear'):
    return data.interpolate(method=method, axis=1, limit_direction='both')

def filter_noise_and_fill(data, window_size=60, threshold_factor=0.75, median_window=40):
    data = data.replace(0, np.nan)
    data = data.where((data >= 250) & (data <= 520), np.nan)
    data = data.apply(lambda x: pd.Series(median_filter(x, size=median_window)), axis=1)
    
    rolling_mean = data.T.rolling(window=window_size, min_periods=1, center=True).mean().T
    rolling_std = data.T.rolling(window=window_size, min_periods=1, center=True).std().T
    rolling_std = np.where(data.isna(), np.nan, rolling_std)

    upper_threshold = rolling_mean + threshold_factor * rolling_std
    lower_threshold = rolling_mean - threshold_factor * rolling_std

    data_filtered = data.where((data >= lower_threshold) & (data <= upper_threshold), np.nan)
    data_filtered = interpolate_nan_values(data_filtered)

    return data_filtered, rolling_mean, upper_threshold, lower_threshold

def second_noise_filter(data, window_size=9, threshold_factor=0.5, median_window=5):
    data = data.apply(lambda x: pd.Series(median_filter(x, size=median_window)), axis=1)
    rolling_mean = data.T.rolling(window=window_size+15, min_periods=1, center=True).mean().T
    rolling_std = data.T.rolling(window=window_size+15, min_periods=1, center=True).std().T

    upper_threshold = rolling_mean + threshold_factor * rolling_std
    lower_threshold = rolling_mean - (threshold_factor * 1.25) * rolling_std

    data_filtered = data.where((data >= lower_threshold) & (data <= upper_threshold), np.nan)
    data_interpolated = interpolate_nan_values(data_filtered)

    data_smoothed = pd.DataFrame([savgol_filter(row, window_size, polyorder=3) for row in data_interpolated.values], 
                                  index=data_interpolated.index, columns=data_interpolated.columns)

    return data_smoothed, rolling_mean, upper_threshold, lower_threshold

def save_filtered_data(data, original_path, output_directory):
    # Generiere den neuen Dateipfad im Ausgabeordner
    base_name = os.path.basename(original_path)
    output_path = os.path.join(output_directory, base_name)
    data.to_csv(output_path, sep=";", index=False, header=False)
    print(f"Gespeichert: {output_path}")

def process_all_files(file_directory, output_directory):
    all_measurements = load_data(file_directory)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Erstellen des Ausgabeordners, falls er nicht existiert
    
    processed_data = []
    
    for df, file_path in all_measurements:
        df_filtered, rolling_mean_first, upper_threshold_first, lower_threshold_first = filter_noise_and_fill(df)
        df_second_filtered, rolling_mean_second, upper_threshold_second, lower_threshold_second = second_noise_filter(df_filtered)

        # Speichere die bearbeiteten Daten
        save_filtered_data(df_second_filtered, file_path, output_directory)
        
        # Optional: Hinzufügen zur Liste, falls weitere Verarbeitung gewünscht ist
        processed_data.append(df_second_filtered)

    return processed_data

# Beispiel für die Anwendung
file_directory = r"B:\test_files_shift_correct_bigset"
output_directory = r"B:\filtered_output"  # Zielverzeichnis für die gefilterten Dateien
processed_data = process_all_files(file_directory, output_directory)

# Anzeigen des bereinigten DataFrames der ersten Datei
print("Bereinigte Daten (erste Datei):")
print(processed_data[0].head())
