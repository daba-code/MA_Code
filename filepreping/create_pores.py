import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import glob
import random

# Funktion zur Erstellung des Mittelwertmodells aus allen geladenen Nahtprofilen
def create_mean_model(file_directory):
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
    
    mean_model = np.mean(all_profiles, axis=0)
    return mean_model

# Funktion zur Ermittlung des Nahtbereichs in einer Reihe
def find_seam_area(row, window_size=50):
    min_value = np.inf
    min_index = 0
    for start in range(len(row) - window_size + 1):
        window = row[start:start + window_size]
        window_mean = np.mean(window)
        if window_mean < min_value:
            min_value = window_mean
            min_index = start
    return min_index, min_value

# Funktion zur Erzeugung einer Pore innerhalb des Nahtbereichs
def add_pore_to_profile(profile, seam_start, seam_depth, pore_depth, pore_width):
    profile_with_pore = np.copy(profile)
    center = seam_start + pore_width // 2  # Setze die Mitte der Pore
    
    # Füge die Pore mit gleichmäßiger Tiefe ohne weichere Übergänge ein
    for i in range(seam_start, seam_start + pore_width):
        profile_with_pore[i] -= pore_depth * (profile_with_pore[i] / seam_depth)
    return profile_with_pore

# Funktion zur Generierung von Poren in zufälligen Bereichen der Naht mit unterschiedlichen Größen
def add_multiple_pores(mean_model, num_pores=5, pore_profile_span=10):
    rows, cols = mean_model.shape
    pore_model = np.copy(mean_model)
    pore_positions = []  # Speichert die Positionen der Poren zur Markierung

    # Porenparameter-Listen für die verschiedenen Größen und Tiefen
    pore_width_options = [5, 10, 20, 30]  # Kleine, mittlere und große Porenbreiten
    pore_depth_options = [0.05, 0.15, 0.3]  # Geringe, mittlere und hohe Tiefen relativ zur Nahtprofilhöhe

    for _ in range(num_pores):
        start_profile = random.randint(0, rows - pore_profile_span)  # Zufälliger Start für die Pore über mehrere Profile
        end_profile = start_profile + pore_profile_span

        # Zufällige Auswahl von Pore-Breite und -Tiefe
        pore_width = random.choice(pore_width_options)
        pore_depth_factor = random.choice(pore_depth_options)

        for i, profile_idx in enumerate(range(start_profile, end_profile)):
            seam_start, seam_depth = find_seam_area(mean_model[profile_idx])
            adjusted_pore_depth = pore_depth_factor * seam_depth  # Berechnung der Tiefe auf Basis der Naht
            
            # Pore hinzufügen
            pore_model[profile_idx] = add_pore_to_profile(
                pore_model[profile_idx], seam_start, seam_depth, adjusted_pore_depth, pore_width
            )
            # Markiere die Position der Pore
            pore_positions.extend([(profile_idx, j) for j in range(seam_start, seam_start + pore_width)])

    return pore_model, pore_positions

# Hauptprogramm zum Erstellen und Plotten des Modells mit Poren
file_directory = r"B:\filtered_output_NaN_TH\sortiert"
mean_model = create_mean_model(file_directory)
pore_model, pore_positions = add_multiple_pores(mean_model, num_pores=10)

# Visualisierung des Original- und Porenmodells mit Markierung
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(mean_model, cmap='viridis', aspect='auto')
plt.colorbar(label='Height')
plt.title("Original Mean Model")

plt.subplot(1, 2, 2)
plt.imshow(pore_model, cmap='viridis', aspect='auto')
# Markiere die Poren in rot
for row, col in pore_positions:
    plt.plot(col, row, 'r.', markersize=2)
plt.colorbar(label='Height')
plt.title("Mean Model with Simulated Pores")

plt.show()

# Ergebnis speichern
output_path = file_directory + "/mean_model_with_pores.csv"
pd.DataFrame(pore_model).astype(int).to_csv(output_path, sep=";", index=False, header=False)
print(f"Modell mit Poren gespeichert: {output_path}")
