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

# Funktion zum Laden der Testdatei
def load_test_data(test_file, num_profiles, heights_per_profile):
    print(f"Lade Testdatei {test_file}...")
    test_data = pd.read_csv(test_file, header=None, delimiter=';', skip_blank_lines=True).dropna().values
    if test_data.shape != (num_profiles, heights_per_profile):
        raise ValueError(f"Testdatei {test_file} hat nicht die erwartete Form von {num_profiles}x{heights_per_profile}")
    print("Testdatei erfolgreich geladen.")
    return test_data

# Lade die CSV-Dateien und ermittle automatisch die Struktur
verzeichnis = r'B:\temp\train'
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
model_directory = 'B:\\temp_models'
model_filename = os.path.join(model_directory, 'gpr_model.pkl')  # Speichert in B:\temp_models
num_inducing = 2000
kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=5.)

# Überprüfen, ob das Verzeichnis existiert; andernfalls erstellen
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    print(f"Verzeichnis {model_directory} erstellt.")

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

# Verzeichnis und Dateiname zur Testdatei angeben
test_directory = r'B:\temp\test'  # Verzeichnis zur Testdatei
test_filename = 'opt_opt_00146195_Seam_Seam_right__8.pqs.csv'  # Name der Testdatei
# Laden der Testdatei und Berechnung der Vorhersageabweichung
test_file = os.path.join(test_directory, test_filename)
test_data = load_test_data(test_file, num_profiles, heights_per_profile)

# Vorhersage für die Testdatei
X_test, Y_test = np.meshgrid(x, y, indexing='ij')
X_test_flat = X_test.flatten().reshape(-1, 1)
Y_test_flat = Y_test.flatten().reshape(-1, 1)
test_input = np.hstack((X_test_flat, Y_test_flat))

print("Starte Vorhersage für die Testdatei...")
mean_pred, var_pred = model.predict(test_input)
Z_pred_test = mean_pred.reshape(X_test.shape)

# RMSE und R^2 Berechnung für die Testdatei
print("Berechne RMSE und R²-Wert für die Testdatei...")
test_data_flat = test_data.flatten()
Z_pred_test_flat = Z_pred_test.flatten()

rmse_test = np.sqrt(mean_squared_error(test_data_flat, Z_pred_test_flat))
r2_test = r2_score(test_data_flat, Z_pred_test_flat)

print(f"RMSE für die Testdatei: {rmse_test:.4f}")
print(f"R^2 für die Testdatei: {r2_test:.4f}")

# 3D-Plot der tatsächlichen Daten aus der Testdatei
fig = plt.figure(figsize=(18, 8))

# Tatsächliche Testdaten visualisieren
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X_test, Y_test, test_data, cmap="viridis", edgecolor='none')
ax1.set_title("Tatsächliche Höhenwerte (Testdaten)")
ax1.set_xlabel("Profile (X)")
ax1.set_ylabel("Höhenindex (Y)")
ax1.set_zlabel("Höhenwert (Z)")

# Vorhergesagte Daten mit Konfidenzintervall visualisieren
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X_test, Y_test, Z_pred_test, cmap="viridis", edgecolor='none', label="Vorhersage")
ax2.plot_surface(X_test, Y_test, Z_pred_test - 1.96 * np.sqrt(var_pred).reshape(X_test.shape), color='orange', alpha=0.3, edgecolor='none', label="95% Konfidenzintervall Untergrenze")
ax2.plot_surface(X_test, Y_test, Z_pred_test + 1.96 * np.sqrt(var_pred).reshape(X_test.shape), color='orange', alpha=0.3, edgecolor='none', label="95% Konfidenzintervall Obergrenze")
ax2.set_title("Vorhergesagte Höhenwerte mit Konfidenzintervall")
ax2.set_xlabel("Profile (X)")
ax2.set_ylabel("Höhenindex (Y)")
ax2.set_zlabel("Vorhergesagter Höhenwert (Z)")

plt.tight_layout()
plt.show()