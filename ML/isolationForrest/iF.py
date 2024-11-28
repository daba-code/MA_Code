import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 1. CSV-Datei laden
file_path = r"C:\Users\dabac\Desktop\file_csv_test\test.csv"  # Ersetzen Sie dies durch den tatsächlichen Pfad
data_csv = pd.read_csv(file_path, sep=";", header=None)
print(data_csv.head(10))  # Zeigt die ersten 10 Zeilen zur Diagnose

# 2. Daten umformen
data = data_csv.stack().reset_index()
data.columns = ['z', 'x', 'y']  # Benennen Sie die Spalten für z (Reihe), x (Spalte), und y (Zellenwert)

# Konvertieren Sie x und z in numerische Werte, falls sie nicht numerisch sind
data['x'] = data['x'].astype(int)
data['z'] = data['z'].astype(int)

# 3. Isolation Forest trainieren
iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
data['anomaly_score'] = iso_forest.fit_predict(data[['x', 'z', 'y']])
data['is_anomaly'] = data['anomaly_score'] == -1  # -1 zeigt Anomalien an

# 4. Visualisierung der Ergebnisse

# 4.1 3D-Plot mit Scatter für Normaldaten und rote Punkte für Anomalien
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Normaldaten als Scatter darstellen
normal_data = data[~data['is_anomaly']]
scatter = ax.scatter(
    normal_data['x'], normal_data['z'], normal_data['y'],
    c=normal_data['y'], cmap='viridis', label='Normaldaten (Scatter)', alpha=0.8
)

# Anomalien als rote Punkte darstellen
anomaly_data = data[data['is_anomaly']]
ax.scatter(
    anomaly_data['x'], anomaly_data['z'], anomaly_data['y'],
    c='red', label='Anomalien (Punkte)', s=50, zorder=5
)

# Achsenbeschriftung und Titel
ax.set_title('Isolation Forest: Anomalien und Höhenprofil (Normaldaten)', fontsize=14)
ax.set_xlabel('Spalte (x)', fontsize=12)
ax.set_ylabel('Reihe (z)', fontsize=12)
ax.set_zlabel('Höhe (y)', fontsize=12)

# Colorbar für die Normaldaten hinzufügen
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Höhe (y)', fontsize=12)

# Legende und Anzeige
ax.legend()
plt.show()


