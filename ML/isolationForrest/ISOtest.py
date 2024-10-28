import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import glob
import os

# Verzeichnis mit Dateien
file_directory = r"B:\filtered_output_NaN_TH\sortiert\nok"
file_paths = glob.glob(os.path.join(file_directory, "*.csv"))

# Alle Dateien laden und zu einem DataFrame kombinieren
data_frames = []
for file_path in file_paths:
    df = pd.read_csv(file_path, sep=";", header=None)
    data_frames.append(df)

# Gesamt-Daten in ein einziges DataFrame zusammenfügen
data = pd.concat(data_frames, ignore_index=True)

# Isolation Forest mit optimierten Parametern
iso_forest = IsolationForest(
    n_estimators=300,              # Höhere Anzahl an Bäumen
    contamination=0.01,            # Geringerer Anteil an Anomalien
    max_samples=256,               # Probenanzahl pro Baum
    max_features=0.75,             # Teilmenge der Merkmale verwenden
    random_state=42
)

# Daten in ein 2D-Array konvertieren
X = data.values

# Trainieren des Isolation Forest und Vorhersage der Anomalien
predictions = iso_forest.fit_predict(X)

# Ausgabe von Anomalien
anomalies = np.where(predictions == -1)
print("Anzahl gefundener Anomalien:", len(anomalies[0]))

# Visualisierung der Anomalien in der Heatmap
plt.figure(figsize=(10, 6))
plt.imshow(X, cmap='viridis', aspect='auto')
for row in anomalies[0]:
    plt.axhline(y=row, color='red', linestyle='--', alpha=0.3)  # Anomalien markieren
plt.colorbar(label='Height')
plt.title("Isolation Forest Anomalieerkennung")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()
