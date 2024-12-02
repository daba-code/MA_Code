import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import glob

# 1. CSV-Datei laden
file_path = r"C:\Users\dabac\Desktop\file_csv_test\mean_model_with_pores.csv"

# Funktion, um aus jeder Datei die gewünschte Reihe zu laden
def load_row_from_files(file_directory, row_number=100):
    file_paths = glob.glob(f"{file_directory}/*.csv")
    all_data = []
    
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=";", header=None)  # Laden der Datei
        if row_number < len(df):
            row_data = df.iloc[row_number].values  # Gewünschte Reihe (z.B. Reihe 100)
            all_data.append(row_data)  # Speichern der y-Werte (Höhenwerte)
    
    return np.array(all_data)  # Rückgabe der y-Werte als numpy-Array

# Ergebnisse speichern
results = []
data_csv = load_row_from_files(file_path)

# 2. Isolation Forest für jede Spalte ausführen
for column in data_csv.columns:
    y_values = data_csv[column]  # Alle y-Werte der aktuellen Spalte
    z_indices = np.arange(len(y_values))  # Zeilenindex für z

    # Isolation Forest trainieren
    iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    anomaly_scores = iso_forest.fit_predict(y_values.values.reshape(-1, 1))  # Isolation Forest auf Spaltenwerten

    # Speichern der Ergebnisse: Spaltenname, z-Index, y-Wert, Anomalie
    results.extend(
        [
            {"x": column, "z": z, "y": y, "is_anomaly": anomaly == -1}
            for z, y, anomaly in zip(z_indices, y_values, anomaly_scores)
        ]
    )

# Ergebnisse als DataFrame speichern
results_df = pd.DataFrame(results)

# 3. Visualisierung der Ergebnisse (2D)
plt.figure(figsize=(14, 8))

# Normaldaten plotten
normal_data = results_df[~results_df['is_anomaly']]
plt.scatter(normal_data['x'], normal_data['y'], c='blue', label='Normaldaten', alpha=0.7, s=1)

# Anomalien plotten
anomaly_data = results_df[results_df['is_anomaly']]
plt.scatter(anomaly_data['x'], anomaly_data['y'], c='red', label='Anomalien', alpha=0.9, s=5)

# Achsenbeschriftung und Titel
plt.title('Isolation Forest: Anomalien pro Spalte (2D-Ansicht)', fontsize=14)
plt.xlabel('Spalte (x)', fontsize=12)
plt.ylabel('Höhenwert (y)', fontsize=12)

# Legende und Anzeige
plt.legend()
plt.grid(True)
plt.show()
