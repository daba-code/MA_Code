import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GPy
from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm

# 1. CSV-Dateien laden
def load_csv_files(directory):
    file_paths = glob(os.path.join(directory, "*.csv"))
    data_frames = []
    
    for file_path in file_paths:
        df = pd.read_csv(file_path, delimiter=';', header=None, skip_blank_lines=True)
        data_frames.append(df.iloc[100:121])  # Nur Reihen 100 bis 120 laden
        print(f"Datei geladen: {file_path} mit {df.shape[0]} Zeilen und {df.shape[1]} Spalten.")
    
    return data_frames

# 2. Daten vorbereiten
def prepare_data(data_frames):
    """
    X entspricht den Profil-Positionen und y_data enthält die Werte für jedes Profil (Reihe) in den Dateien.
    """
    X = np.arange(383).reshape(-1, 1)  # 383 Positionen entlang des Profils
    y_data = np.array([df.values for df in data_frames])  # Jede Datei als separate y-Reihe
    return X, y_data

# 3. Trainiere das GPR für ein Profil
def train_gpr_for_profile(X, y_profiles, profile_idx):
    """
    Trainiert das GPR-Modell für ein bestimmtes Profil (profile_idx) über alle Dateien.
    """
    kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
    gpr = GPy.models.GPRegression(X, y_profiles[:, profile_idx, :].T.mean(axis=1).reshape(-1, 1), kernel)
    gpr.Gaussian_noise.variance = 0.1

    # Optimierung des Modells
    gpr.optimize(messages=True, max_iters=1000)

    # Vorhersagen für die Testdaten
    y_pred, y_var = gpr.predict(X)
    y_std = np.sqrt(y_var)  # Standardabweichung für Konfidenzintervall

    return {
        "profile_idx": profile_idx,
        "y_pred": y_pred,
        "y_std": y_std
    }

# 4. Plotten der Ergebnisse
def plot_results(X, y_profiles, gpr_results, profile_idx):
    """
    Visualisiert die GPR-Vorhersage, die Varianz, den Mittelwert und die Trainingsdaten.
    """
    plt.figure(figsize=(12, 6))

    # Trainingsdatenpunkte
    y = y_profiles[:, profile_idx]
    for file_idx in range(y_profiles.shape[0]):
        plt.plot(X, y[file_idx], color='blue', alpha=0.2, label='Trainingsdaten' if file_idx == 0 else "")

    # GPR Vorhersage und Unsicherheit
    plt.plot(X, gpr_results["y_pred"], color='black', label='GPy Vorhersage')
    plt.fill_between(
        X.ravel(),
        gpr_results["y_pred"].flatten() - 1.96 * gpr_results["y_std"].flatten(),
        gpr_results["y_pred"].flatten() + 1.96 * gpr_results["y_std"].flatten(),
        alpha=0.2,
        color='gray',
        label='95% Konfidenzintervall'
    )

    plt.title(f"Profil {profile_idx}: GPR-Vorhersage")
    plt.xlabel("Position entlang des Profils")
    plt.ylabel("Wert")
    plt.legend()
    plt.show()

# 5. Verarbeitung aller Profile
def process_all_profiles(X, y_profiles):
    """
    Führt GPR-Modelle auf allen Profilen parallel aus und visualisiert die Ergebnisse.
    """
    results = Parallel(n_jobs=-1)(delayed(train_gpr_for_profile)(X, y_profiles, profile_idx) 
                                  for profile_idx in tqdm(range(y_profiles.shape[1]), desc="Trainiere GPR für Profile"))

    # Ergebnisse für jedes Profil visualisieren
    for gpr_res in results:
        plot_results(X, y_profiles, gpr_res, gpr_res["profile_idx"])

# 6. Hauptprogramm
def main(directory):
    data_frames = load_csv_files(directory)
    X, y_profiles = prepare_data(data_frames)
    process_all_profiles(X, y_profiles)

# Beispielaufruf
directory = r'B:\filtered_output\gekuerzt\aligned_files\processed_files\gekuerzt'
main(directory)
