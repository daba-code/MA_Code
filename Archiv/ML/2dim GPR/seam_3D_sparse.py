import numpy as np
import GPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed
import os
import pandas as pd
from tqdm import tqdm
import pickle

# Funktion zum Speichern des GPR-Modells
def save_gpr_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model.param_array, file)
    print(f"Modell wurde unter {filename} gespeichert.")

# Funktion zum Laden des GPR-Modells
def load_gpr_model(input_data, target_data, kernel, num_inducing, filename):
    model = GPy.models.SparseGPRegression(input_data, target_data, kernel, num_inducing=num_inducing)
    with open(filename, 'rb') as file:
        param_array = pickle.load(file)
    model.param_array[:] = param_array
    model.update_model(False)
    print(f"Modell wurde aus {filename} geladen.")
    return model

# Funktion zum Laden der CSV-Dateien und Ermittlung der Datenstruktur
def load_height_data(directory):
    print("Starte das Laden der CSV-Dateien...")
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError("Keine CSV-Dateien im Verzeichnis gefunden")
    
    # Ermittlung der Struktur anhand der ersten Datei
    first_file_path = os.path.join(directory, csv_files[0])
    first_data = pd.read_csv(first_file_path, header=None, delimiter=';', skip_blank_lines=True).dropna().values
    num_profiles, heights_per_profile = first_data.shape
    print(f"Ermittelte Struktur: {num_profiles} Profile und {heights_per_profile} Höhenwerte pro Profil")

    # CSV-Dateien laden und in Liste speichern
    datasets = [first_data]
    for file in tqdm(csv_files[1:], desc="Lade CSV-Dateien", unit=" Datei"):
        data = pd.read_csv(os.path.join(directory, file), header=None, delimiter=';', skip_blank_lines=True).dropna().values
        if data.shape != (num_profiles, heights_per_profile):
            raise ValueError(f"Datei {file} hat nicht die erwartete Form von {num_profiles}x{heights_per_profile}")
        datasets.append(data)
    
    print("Alle CSV-Dateien wurden erfolgreich geladen.")
    return datasets, num_profiles, heights_per_profile

# Lade die CSV-Dateien und ermittle automatisch die Struktur
verzeichnis = r'B:\temp'
datasets, num_profiles, heights_per_profile = load_height_data(verzeichnis)

# Erzeugen des Gitters für die Profile (x) und Höhen (y)
print("Erzeuge das Gitter für Profile und Höhenwerte...")
x = np.arange(num_profiles)
y = np.arange(heights_per_profile)
X, Y = np.meshgrid(x, y, indexing='ij')

# Erstellen der flachen Arrays für das Training
print("Erstelle flache Arrays für das Training...")
X_flat = np.tile(X.flatten(), len(datasets)).reshape(-1, 1)
Y_flat = np.tile(Y.flatten(), len(datasets)).reshape(-1, 1)
Z_flat = np.concatenate([data.flatten() for data in datasets]).reshape(-1, 1)

print("X_flat shape:", X_flat.shape)
print("Y_flat shape:", Y_flat.shape)
print("Z_flat shape:", Z_flat.shape)

# Zusammengefügte Eingabedaten und Ziel-Daten
input_data = np.hstack((X_flat, Y_flat))
target_data = Z_flat
print("Zusammengeführte Eingabedaten (input_data) und Ziel-Daten (target_data) erstellt.")

# Modelldefinition und -training oder Laden eines gespeicherten Modells
model_filename = "gpr_model.pkl"
num_inducing = 2000
kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=5.)

# Überprüfen, ob das Modell bereits gespeichert ist
if os.path.exists(model_filename):
    print(f"Gespeichertes Modell {model_filename} gefunden. Lade das Modell...")
    model = load_gpr_model(input_data, target_data, kernel, num_inducing, model_filename)
else:
    print("Kein gespeichertes Modell gefunden. Starte das Training...")
    model = GPy.models.SparseGPRegression(input_data, target_data, kernel, num_inducing=num_inducing)
    #model.optimize(messages=True)
    print("Modelltraining abgeschlossen. Speichere das Modell...")
    save_gpr_model(model, model_filename)

# Schritt 3: Vorhersage mit paralleler Verarbeitung
X_pred, Y_pred = np.meshgrid(x, y, indexing='ij')
X_pred_flat = X_pred.flatten().reshape(-1, 1)
Y_pred_flat = Y_pred.flatten().reshape(-1, 1)
new_input = np.hstack((X_pred_flat, Y_pred_flat))

# Aufteilen des Eingabegitters in Batches für parallele Verarbeitung
batch_size = 3000
input_batches = [new_input[i:i+batch_size] for i in range(0, new_input.shape[0], batch_size)]
print(f"Aufgeteilt in {len(input_batches)} Batches für parallele Verarbeitung.")

# Parallele Berechnung der Vorhersagen mit joblib und Fortschrittsanzeige
print("Starte die Batch-Verarbeitung für die Vorhersage...")
results = []
for i, batch in enumerate(tqdm(input_batches, desc="Batch-Verarbeitung", unit=" Batch", total=len(input_batches))):
    results.append(model.predict(batch))
    print(f"Batch {i+1}/{len(input_batches)} verarbeitet.")

print("Batch-Verarbeitung abgeschlossen. Kombiniere Ergebnisse...")

# Ergebnisse zusammenführen
Z_pred = np.concatenate([res[0] for res in results]).reshape(X_pred.shape)
Z_pred_var = np.concatenate([res[1] for res in results]).reshape(X_pred.shape)
Z_pred_std = np.sqrt(Z_pred_var)

# Konfidenzintervalle berechnen
confidence_interval_lower = Z_pred - 1.96 * Z_pred_std
confidence_interval_upper = Z_pred + 1.96 * Z_pred_std

print("Vorhersage abgeschlossen. Starte Visualisierung...")

# Schritt 4: Visualisierung der 3D-Struktur, der GPR-Vorhersage und der Varianz
fig = plt.figure(figsize=(24, 10))

# Originaldaten der ersten Datei
ax1 = fig.add_subplot(151, projection='3d')
ax1.plot_surface(X, Y, datasets[0], cmap="viridis", edgecolor='none')
ax1.set_title("Originale Profilstruktur (Datei 1)")
ax1.set_xlabel("Profile (X)")
ax1.set_ylabel("Höhenindex (Y)")
ax1.set_zlabel("Höhenwert (Z)")

# Weitere Originaldaten visualisieren, falls vorhanden
if len(datasets) > 1:
    ax2 = fig.add_subplot(152, projection='3d')
    ax2.plot_surface(X, Y, datasets[1], cmap="viridis", edgecolor='none')
    ax2.set_title("Originale Profilstruktur (Datei 2)")
    ax2.set_xlabel("Profile (X)")
    ax2.set_ylabel("Höhenindex (Y)")
    ax2.set_zlabel("Höhenwert (Z)")

if len(datasets) > 2:
    ax3 = fig.add_subplot(153, projection='3d')
    ax3.plot_surface(X, Y, datasets[2], cmap="viridis", edgecolor='none')
    ax3.set_title("Originale Profilstruktur (Datei 3)")
    ax3.set_xlabel("Profile (X)")
    ax3.set_ylabel("Höhenindex (Y)")
    ax3.set_zlabel("Höhenwert (Z)")

# Vorhersage visualisieren
ax4 = fig.add_subplot(154, projection='3d')
ax4.plot_surface(X_pred, Y_pred, Z_pred, cmap="viridis", edgecolor='none')
ax4.plot_surface(X_pred, Y_pred, confidence_interval_lower, color='orange', alpha=0.3, edgecolor='none')
ax4.plot_surface(X_pred, Y_pred, confidence_interval_upper, color='orange', alpha=0.3, edgecolor='none')
ax4.set_title("GPR Vorhersage mit Konfidenzintervall")
ax4.set_xlabel("Profile (X)")
ax4.set_ylabel("Höhenindex (Y)")
ax4.set_zlabel("Vorhergesagter Höhenwert (Z)")

# Varianz als separate Fläche visualisieren
ax5 = fig.add_subplot(155, projection='3d')
ax5.plot_surface(X_pred, Y_pred, Z_pred_std**2, cmap="inferno", edgecolor='none')
ax5.set_title("Varianz der Vorhersage")
ax5.set_xlabel("Profile (X)")
ax5.set_ylabel("Höhenindex (Y)")
ax5.set_zlabel("Varianz")

plt.tight_layout()
plt.show()
print("Visualisierung abgeschlossen.")

# RMSE und R^2 berechnen
print("Berechne RMSE und R²-Wert für die Vorhersagegenauigkeit...")
Z1_flat = datasets[0].flatten()
Z_pred_flat = Z_pred.flatten()
rmse = np.sqrt(mean_squared_error(Z1_flat, Z_pred_flat))
r2 = r2_score(Z1_flat, Z_pred_flat)

print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")
print("Berechnung abgeschlossen.")
