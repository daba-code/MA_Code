from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, medfilt
import glob
import matplotlib.pyplot as plt

def load_data(file_directory):
    """
    Lädt alle CSV-Dateien in einem Verzeichnis und speichert sie in einer Liste von DataFrames.
    """
    file_paths = glob.glob(f"{file_directory}/*.csv")
    all_measurements = []

    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=";", header=None)
        all_measurements.append(df)
    
    return all_measurements

def interpolate_nan_values(data, method='linear'):
    """
    Interpoliert NaN-Werte im DataFrame.
    
    Parameter:
    - data: Pandas DataFrame mit NaN-Werten.
    - method: Methode zur Interpolation ('linear', 'polynomial', etc.).
    
    Rückgabe:
    - data: DataFrame mit interpolierten Werten.
    """
    return data.interpolate(method=method, axis=1, limit_direction='both')

def filter_noise_and_fill(data, window_size=30, threshold_factor=3):
    """
    Behandelt 0-Werte als NaN und definiert Rauschen anhand des gleitenden Mittelwerts.
    
    Parameter:
    - data: Pandas DataFrame.
    - window_size: Größe des Fensters für den gleitenden Mittelwert.
    - threshold_factor: Faktor zur Skalierung des Schwellenwerts.

    Rückgabe:
    - data_filtered: DataFrame mit gefilterten Werten.
    """
    # 0-Werte als NaN behandeln und Werte außerhalb von 200 bis 540 ebenfalls auf NaN setzen
    data = data.replace(0, np.nan)
    data = data.where((data >= 200) & (data <= 540), np.nan)

    # Berechnung des gleitenden Mittels und der Standardabweichung
    rolling_mean = data.T.rolling(window=window_size, min_periods=1, center=True).mean().T
    rolling_std = data.T.rolling(window=window_size, min_periods=1, center=True).std().T

    # Dynamische Schwellenwerte
    upper_threshold = rolling_mean + threshold_factor * rolling_std
    lower_threshold = rolling_mean - threshold_factor * rolling_std

    # Markiere Werte außerhalb des dynamischen Bereichs als NaN
    data_filtered = data.where((data >= lower_threshold) & (data <= upper_threshold), np.nan)

    # Interpolieren der NaN-Werte
    data_filtered = interpolate_nan_values(data_filtered)

    return data_filtered, rolling_mean, upper_threshold, lower_threshold

def scale_with_standard_scaler(data):
    """
    Wendet den StandardScaler auf jedes Profil (jede Zeile) an, sodass der Mittelwert 0 und die Standardabweichung 1 beträgt.
    """
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data.T).T, columns=data.columns)
    
    return data_scaled

def visualize_filtering(data_original, data_filtered, rolling_mean, upper_threshold, lower_threshold):
    """
    Visualisiert jedes Profil vor und nach der Filterung im interaktiven Modus,
    einschließlich dynamischer Schwellenwerte.
    """
    plt.ion()  # Interaktiver Modus für dynamisches Plotten
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, (original_row, filtered_row, mean_row, upper_row, lower_row) in enumerate(zip(
        data_original.iterrows(), data_filtered.iterrows(), 
        rolling_mean.iterrows(), upper_threshold.iterrows(), lower_threshold.iterrows())):
        
        ax.clear()  # Bereinigen des Plots für das nächste Profil
        
        # Originaldaten in Rot plotten
        ax.plot(original_row[1].index, original_row[1].values, label=f"Original Profil {idx + 1}", color='red', alpha=0.6)
        
        # Gefilterte Daten in Blau plotten
        ax.plot(filtered_row[1].index, filtered_row[1].values, label=f"Gefiltertes Profil {idx + 1}", color='blue', alpha=0.8)
        
        # Dynamische Schwellenwerte und gleitender Mittelwert hinzufügen
        ax.plot(mean_row[1].index, mean_row[1].values, label="Gleitender Mittelwert", color="green", linestyle="--")
        ax.plot(upper_row[1].index, upper_row[1].values, label="Oberer Schwellenwert", color="purple", linestyle=":")
        ax.plot(lower_row[1].index, lower_row[1].values, label="Unterer Schwellenwert", color="purple", linestyle=":")

        ax.set_title(f"Original vs. Gefiltertes Profil {idx + 1}")
        ax.set_xlabel("Position entlang des Profils")
        ax.set_ylabel("Wert")
        ax.legend()
        
        plt.pause(2)  # Plot 2 Sekunden anzeigen

    plt.ioff()  # Interaktiven Modus ausschalten
    plt.show()  # Sicherstellen, dass der letzte Plot sichtbar bleibt

def process_all_files(file_directory):
    # Laden aller Daten
    all_measurements = load_data(file_directory)
    processed_data = []
    
    # Filterung und Visualisierung für jede Datei
    for df in all_measurements:
        df_filtered, rolling_mean, upper_threshold, lower_threshold = filter_noise_and_fill(df)  # Rückgabewerte anpassen
        visualize_filtering(df, df_filtered, rolling_mean, upper_threshold, lower_threshold)  # Übergabe der zusätzlichen Argumente
        processed_data.append(df_filtered)

    return processed_data

# Beispiel für die Anwendung
file_directory = r"B:\test_files_shift_correct"
processed_data = process_all_files(file_directory)

# Anzeigen des bereinigten DataFrames der ersten Datei
print("Bereinigte Daten (erste Datei):")
print(processed_data[0].head())
