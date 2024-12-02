import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 1. CSV-Datei laden
file_path = r"C:\Users\dabac\Desktop\file_csv_test\test.csv"  # Ersetzen Sie dies durch den tatsächlichen Pfad
data_csv = pd.read_csv(file_path, sep=";", header=None)

# Ergebnisse speichern
results = []

# 2. Isolation Forest für jede Reihe ausführen
for index, row in data_csv.iterrows():
    x_indices = np.arange(len(row))  # Spaltenindex für x
    y_values = row.values  # Alle y-Werte der aktuellen Reihe

    # Isolation Forest trainieren
    iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    anomaly_scores = iso_forest.fit_predict(y_values.reshape(-1, 1))  # Isolation Forest auf Reihenwerten

    # Speichern der Ergebnisse: Zeilenname, x-Index, y-Wert, Anomalie
    results.extend(
        [
            {"z": index, "x": x, "y": y, "is_anomaly": anomaly == -1}
            for x, y, anomaly in zip(x_indices, y_values, anomaly_scores)
        ]
    )

# Ergebnisse als DataFrame speichern
results_df = pd.DataFrame(results)

# 3. Visualisierung der Ergebnisse (3D-Scatterplot)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Normaldaten plotten
normal_data = results_df[~results_df['is_anomaly']]
ax.scatter(normal_data['x'], normal_data['z'], normal_data['y'], c='blue', label='Normaldaten', alpha=0.7)

# Anomalien plotten
anomaly_data = results_df[results_df['is_anomaly']]
ax.scatter(anomaly_data['x'], anomaly_data['z'], anomaly_data['y'], c='red', label='Anomalien', s=50, zorder=5)

# Achsenbeschriftung und Titel
ax.set_title('Isolation Forest: Anomalien pro Reihe', fontsize=14)
ax.set_xlabel('Spalte (x)', fontsize=12)
ax.set_ylabel('Reihe (z)', fontsize=12)
ax.set_zlabel('Höhe (y)', fontsize=12)

# Legende und Anzeige
ax.legend()
plt.show()
