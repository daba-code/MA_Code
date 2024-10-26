from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, medfilt
import glob
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

def load_data(file_directory):
    file_paths = glob.glob(f"{file_directory}/*.csv")
    all_measurements = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=";", header=None)
        all_measurements.append(df)
    return all_measurements

def interpolate_nan_values(data, method='linear'):
    return data.interpolate(method=method,axis=1, limit_direction='both')

def filter_noise_and_fill(data, window_size=60, threshold_factor=0.75, median_window=40):
    data = data.replace(0, np.nan)
    data = data.where((data >= 270) & (data <= 520), np.nan)

    # Anwenden des Median-Filters zur Reduktion von Ausreißern
    data = data.apply(lambda x: pd.Series(median_filter(x, size=median_window)), axis=1)

    rolling_mean = data.T.rolling(window=window_size, min_periods=1, center=True).mean().T
    # Berechnung der Standardabweichung, aber nur für nicht NaN-Werte
    rolling_std = data.T.rolling(window=window_size, min_periods=1, center=True).std().T
    rolling_std = np.where(data.isna(), np.nan, rolling_std)  # NaN an den gleichen Stellen wie in den Daten

    upper_threshold = rolling_mean + threshold_factor * rolling_std
    lower_threshold = rolling_mean - (threshold_factor * 1) * rolling_std

    data_filtered = data.where((data >= lower_threshold) & (data <= upper_threshold), np.nan)
    data_filtered = interpolate_nan_values(data_filtered)

    return data_filtered, rolling_mean, upper_threshold, lower_threshold

def second_noise_filter(data, window_size=9, threshold_factor=0.5, median_window=5):

    # Anwenden des Median-Filters zur Reduktion von Ausreißern
    data = data.apply(lambda x: pd.Series(median_filter(x, size=median_window)), axis=1)
    # Berechnung des gleitenden Mittels und der Standardabweichung
    rolling_mean = data.T.rolling(window=window_size+15, min_periods=1, center=True).mean().T
    rolling_std = data.T.rolling(window=window_size+15, min_periods=1, center=True).std().T

    # Dynamische Schwellenwerte für die zweite Filterung
    upper_threshold = rolling_mean + (threshold_factor * rolling_std)  # Anpassung
    lower_threshold = rolling_mean - (threshold_factor * 1.25) * rolling_std

    # Markiere Ausreißer als NaN
    data_filtered = data.where((data >= lower_threshold) & (data <= upper_threshold), np.nan)

    # Interpolation der NaN-Werte
    data_interpolated = interpolate_nan_values(data_filtered)

    # Gleitenden Mittelwert oder andere Filtermethoden anwenden
    data_smoothed = pd.DataFrame([savgol_filter(row, window_size, polyorder=3) for row in data_interpolated.values], 
                                  index=data_interpolated.index, columns=data_interpolated.columns)

    return data_smoothed, rolling_mean, upper_threshold, lower_threshold

def scale_with_standard_scaler(data):
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data.T).T, columns=data.columns)
    return data_scaled

def visualize_filtering(data_original, data_first_filtered, rolling_mean_first, upper_threshold_first, lower_threshold_first,
                        data_second_filtered, rolling_mean_second, upper_threshold_second, lower_threshold_second):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, (original_row, first_filtered_row, mean_row_first, upper_row_first, lower_row_first,
               second_filtered_row, mean_row_second, upper_row_second, lower_row_second) in enumerate(zip(
        data_original.iterrows(), data_first_filtered.iterrows(), 
        rolling_mean_first.iterrows(), upper_threshold_first.iterrows(), lower_threshold_first.iterrows(),
        data_second_filtered.iterrows(), rolling_mean_second.iterrows(), upper_threshold_second.iterrows(), lower_threshold_second.iterrows())):
        
        ax.clear()
        ax.plot(original_row[1].index, original_row[1].values, label=f"Original Profil {idx + 1}", color='red', alpha=0.6)
        ax.plot(first_filtered_row[1].index, first_filtered_row[1].values, label=f"Erste Filterung {idx + 1}", color='blue', alpha=0.8)
        ax.plot(second_filtered_row[1].index, second_filtered_row[1].values, label=f"Zweite Filterung {idx + 1}", color='orange', alpha=0.8)
        ax.plot(mean_row_first[1].index, mean_row_first[1].values, label="Gleitender Mittelwert (1. Filterung)", color="green", linestyle="--")
        ax.plot(upper_row_first[1].index, upper_row_first[1].values, label="Oberer Schwellenwert (1. Filterung)", color="purple", linestyle=":")
        ax.plot(lower_row_first[1].index, lower_row_first[1].values, label="Unterer Schwellenwert (1. Filterung)", color="purple", linestyle=":")

        ax.plot(mean_row_second[1].index, mean_row_second[1].values, label="Gleitender Mittelwert (2. Filterung)", color="lightgreen", linestyle="--")
        ax.plot(upper_row_second[1].index, upper_row_second[1].values, label="Oberer Schwellenwert (2. Filterung)", color="darkviolet", linestyle=":")
        ax.plot(lower_row_second[1].index, lower_row_second[1].values, label="Unterer Schwellenwert (2. Filterung)", color="darkviolet", linestyle=":")

        ax.set_title(f"Original vs. Gefiltertes Profil {idx + 1}")
        ax.set_xlabel("Position entlang des Profils")
        ax.set_ylabel("Wert")
        ax.legend()
        
        plt.pause(3)  # Plot 2 Sekunden anzeigen

    plt.ioff()

def process_all_files(file_directory):
    all_measurements = load_data(file_directory)
    processed_data = []
    
    for df in all_measurements:
        df_filtered, rolling_mean_first, upper_threshold_first, lower_threshold_first = filter_noise_and_fill(df)
        df_second_filtered, rolling_mean_second, upper_threshold_second, lower_threshold_second = second_noise_filter(df_filtered)

        visualize_filtering(df, df_filtered, rolling_mean_first, upper_threshold_first, lower_threshold_first,
                            df_second_filtered, rolling_mean_second, upper_threshold_second, lower_threshold_second)
        processed_data.append(df_second_filtered)

    return processed_data

# Beispiel für die Anwendung
file_directory = r"B:\test_files_shift_correct_bigset"
processed_data = process_all_files(file_directory)

# Anzeigen des bereinigten DataFrames der ersten Datei
print("Bereinigte Daten (erste Datei):")
print(processed_data[0].head())
