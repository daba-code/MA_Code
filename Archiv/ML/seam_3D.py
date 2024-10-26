import numpy as np
import pandas as pd
import GPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed
import glob

# Laden der CSV-Dateien und Bestimmen der minimalen Zeilenanzahl
def load_csv_data(file_pattern):
    file_paths = glob.glob(file_pattern)  # Alle passenden Dateien laden
    data = []
    min_profiles = float('inf')  # Startwert für die minimale Anzahl an Profilen
    heights_per_profile = None

    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=";", header=None, dtype='int16')  # CSV-Dateien ohne Header, dtype int16
        if df.shape[0] < min_profiles:
            min_profiles = df.shape[0]  # Aktualisieren der minimalen Anzahl an Profilen
        if heights_per_profile is None:
            heights_per_profile = df.shape[1]  # Anzahl der Spalten aus der ersten Datei übernehmen
        data.append(df.values.astype('int16'))  # Ensure data is int16
    
    # Kürzen aller Dateien auf die minimale Anzahl an Profilen
    data = [d[:min_profiles, :heights_per_profile].astype('int16') for d in data]
    
    return data, min_profiles, heights_per_profile

# Pfad zu den CSV-Dateien anpassen
file_pattern = r'B:\temp - Copy\*.csv'
data, num_profiles, heights_per_profile = load_csv_data(file_pattern)

# Prüfen, ob Daten geladen wurden
if not data:
    raise ValueError("Keine CSV-Dateien gefunden. Bitte prüfen Sie den Dateipfad oder die Dateierweiterung.")

# Kombiniere die Daten aus den CSV-Dateien entlang der Z-Achse
X, Y = np.meshgrid(np.arange(num_profiles, dtype='int16'), np.arange(heights_per_profile, dtype='int16'), indexing='ij')
Z_flat = np.concatenate([d.flatten() for d in data]).reshape(-1, 1).astype('int16')

# Erstellen der Input-Daten für GPR
X_flat = np.tile(X.flatten(), len(data)).reshape(-1, 1).astype('int16')  # Profile (x)
Y_flat = np.tile(Y.flatten(), len(data)).reshape(-1, 1).astype('int16')  # Höhenwerte (y)
input_data = np.hstack((X_flat, Y_flat)).astype('float32')  # Cast to float32 for GPR input
target_data = Z_flat.astype('float32')                      # Cast to float32 for GPR target

# Normales GPR-Modell einrichten und optimieren
kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=5.)
model = GPy.models.GPRegression(input_data, target_data, kernel)

# Modelloptimierung
model.optimize(messages=True)

# Schritt 3: Vorhersage mit paralleler Verarbeitung
X_pred, Y_pred = np.meshgrid(np.arange(num_profiles, dtype='int16'), np.arange(heights_per_profile, dtype='int16'), indexing='ij')
X_pred_flat = X_pred.flatten().reshape(-1, 1).astype('float32')
Y_pred_flat = Y_pred.flatten().reshape(-1, 1).astype('float32')
new_input = np.hstack((X_pred_flat, Y_pred_flat))

# Funktion für parallele Vorhersageberechnung
def predict_in_batches(input_batch):
    return model.predict(input_batch)

# Aufteilen des Eingabegitters in Batches für parallele Verarbeitung
batch_size = 2000
input_batches = [new_input[i:i+batch_size] for i in range(0, new_input.shape[0], batch_size)]

# Parallele Berechnung der Vorhersagen mit joblib
results = Parallel(n_jobs=-1)(delayed(predict_in_batches)(batch) for batch in input_batches)

# Ergebnisse zusammenführen
Z_pred = np.concatenate([res[0] for res in results]).reshape(X_pred.shape)
Z_pred_var = np.concatenate([res[1] for res in results]).reshape(X_pred.shape)
Z_pred_std = np.sqrt(Z_pred_var)

# Konfidenzintervalle berechnen (95% Konfidenzintervall)
confidence_interval_lower = Z_pred - 1.96 * Z_pred_std
confidence_interval_upper = Z_pred + 1.96 * Z_pred_std

# Schritt 4: Visualisierung der 3D-Struktur, der GPR-Vorhersage und der Varianz
fig = plt.figure(figsize=(24, 10))

# Originaldaten der ersten drei Dateien visualisieren
for i in range(min(3, len(data))):
    ax = fig.add_subplot(1, 5, i+1, projection='3d')
    ax.plot_surface(X, Y, data[i].astype('float32'), cmap="viridis", edgecolor='none')
    ax.set_title(f"Originale Profilstruktur (Datei {i+1})")
    ax.set_xlabel("Profile (X)")
    ax.set_ylabel("Höhenindex (Y)")
    ax.set_zlabel("Höhenwert (Z)")

# GPR-Vorhersage visualisieren mit Mean und Konfidenzintervall
ax4 = fig.add_subplot(1, 5, 4, projection='3d')
ax4.plot_surface(X_pred, Y_pred, Z_pred, cmap="viridis", edgecolor='none')
ax4.plot_surface(X_pred, Y_pred, confidence_interval_lower, color='orange', alpha=0.3, edgecolor='none')
ax4.plot_surface(X_pred, Y_pred, confidence_interval_upper, color='orange', alpha=0.3, edgecolor='none')
ax4.set_title("GPR Vorhersage mit Konfidenzintervall")
ax4.set_xlabel("Profile (X)")
ax4.set_ylabel("Höhenindex (Y)")
ax4.set_zlabel("Vorhergesagter Höhenwert (Z)")

# Varianz als separate Fläche visualisieren
ax5 = fig.add_subplot(1, 5, 5, projection='3d')
ax5.plot_surface(X_pred, Y_pred, Z_pred_std**2, cmap="inferno", edgecolor='none')
ax5.set_title("Varianz der Vorhersage")
ax5.set_xlabel("Profile (X)")
ax5.set_ylabel("Höhenindex (Y)")
ax5.set_zlabel("Varianz")

plt.tight_layout()
plt.show()

# RMSE und R^2 berechnen zwischen der ersten Datei und der Vorhersage
Z1_flat = data[0].flatten().astype('float32')
Z_pred_flat = Z_pred.flatten()
rmse = np.sqrt(mean_squared_error(Z1_flat, Z_pred_flat))
r2 = r2_score(Z1_flat, Z_pred_flat)

print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")
