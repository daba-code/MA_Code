import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Funktion zum Laden und Formatieren der Dateien
def load_and_format_files(file_directory):
    all_profiles = []
    max_rows = 0
    file_paths = glob.glob(f"{file_directory}/*.csv")

    # Lade Dateien und gleiche die Anzahl der Reihen an
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=";", header=None)
        max_rows = max(max_rows, df.shape[0])
        all_profiles.append(df.values)
    
    # Fülle Profile auf gleiche Anzahl an Reihen auf
    for i in range(len(all_profiles)):
        rows_missing = max_rows - all_profiles[i].shape[0]
        if rows_missing > 0:
            all_profiles[i] = np.vstack([all_profiles[i], np.full((rows_missing, all_profiles[i].shape[1]), all_profiles[i][-1])])

    return all_profiles, file_paths

# Funktion zur Visualisierung der einzelnen Dateien als Heatmap
def visualize_all_profiles(all_profiles, file_paths):
    for i, profile in enumerate(all_profiles):
        plt.figure(figsize=(10, 6))
        plt.imshow(profile, cmap="nipy_spectral", aspect='auto')
        plt.colorbar(label='Height')
        plt.title(f"Heatmap for File: {file_paths[i]}")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()

# Hauptprogramm zum Erstellen und Plotten aller Dateien
file_directory = r"B:\filtered_output_NaN_TH\sortiert\nok"
all_profiles, file_paths = load_and_format_files(file_directory)
visualize_all_profiles(all_profiles, file_paths)
