import pandas as pd
import numpy as np
import glob
import GPy
import matplotlib.pyplot as plt

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

# Funktion zum Berechnen des Baseline-Modells (Mittelwert der y-Werte für jeden x-Wert)
def calculate_baseline_model(all_data):
    # Berechnung des Mittelwerts für jede Spalte (x-Wert) unter Ignorieren von NaNs
    baseline_means = np.nanmean(all_data, axis=0)
    return baseline_means

# Funktion zum Filtern und Entfernen von NaNs
def preprocess_data_for_gpr(x_positions, all_data):
    X_train_combined = []
    y_train_combined = []

    # Durch jede Datei (jede Zeile von all_data) iterieren
    for i in range(all_data.shape[0]):
        y_values = all_data[i, :]  # y-Werte der jeweiligen Datei (Reihe)
        valid_indices = ~np.isnan(y_values)  # Gültige Indizes (wo y kein NaN ist)
        X_train_combined.append(x_positions[valid_indices].reshape(-1, 1))  # Gültige x-Werte
        y_train_combined.append(y_values[valid_indices].reshape(-1, 1))  # Gültige y-Werte
    
    # Kombinieren aller x- und y-Werte aus den verschiedenen Dateien
    X_train_combined = np.vstack(X_train_combined)
    y_train_combined = np.vstack(y_train_combined)
    
    return X_train_combined, y_train_combined

# Beispiel: Verwenden des GPR-Modells auf den geladenen Daten
def apply_gpr(file_directory, row_number=200):
    # Lade die Daten für die spezifizierte Reihe (z.B. Reihe 100)
    all_data = load_row_from_files(file_directory, row_number=row_number)
    
    # x-Positionen: Es gibt 383 Spalten (x-Werte), entsprechend den x-Positionen in den Dateien
    x_positions = np.arange(383)
    
    # Daten für GPR vorverarbeiten (entfernt NaNs)
    X_train, y_train = preprocess_data_for_gpr(x_positions, all_data)
    
    # GPR-Modell erstellen und trainieren
    kernel = GPy.kern.Matern32(input_dim=1, variance=1, lengthscale=1, ARD=True)

    kernel.variance.constrain_bounded(0.1, 1200)  # Variance zwischen 0.1 und 100
    kernel.lengthscale.constrain_bounded(0.1, 40)  # Lengthscale zwischen 0.1 und 50
    
    model = GPy.models.GPRegression(X_train, y_train, kernel)
    #model.Gaussian_noise.variance.constrain_bounded(10, 100)

    # Optimierung des Modells
    model.optimize(messages=True)
    
    # Vorhersage für alle x-Werte (einschließlich der NaN-Werte)
    X_test = x_positions.reshape(-1, 1)  # Vorhersagen für alle x-Werte (0 bis 382)
    y_pred, y_var = model.predict(X_test)
    print(model)
    # Rückgabe von X_train, y_train, X_test, y_pred, y_var für das Plotten
    return X_train, y_train, X_test, y_pred, y_var, all_data

# Plotten der Vorhersagen, Konfidenzintervall, Trainingsdaten und des Baseline-Modells
def plot_gpr_with_baseline(X_train, y_train, X_test, y_pred, y_var, baseline_means, all_data):
    # Berechnen des Konfidenzintervalls (95% Konfidenzintervall)
    y_std = np.sqrt(y_var)
    lower_bound = y_pred - 1.96 * y_std
    upper_bound = y_pred + 1.96 * y_std

    # Plotten der GPR-Vorhersagen und des Konfidenzintervalls
    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_pred, 'r-', label='GPR Vorhersage')  # GPR Vorhersage
    plt.fill_between(X_test.flatten(), lower_bound.flatten(), upper_bound.flatten(), 
                     alpha=0.2, color='gray', label='95% Konfidenzintervall')  # Konfidenzintervall
    
    # Farben für die unterschiedlichen Trainingsdateien
    colors = plt.cm.get_cmap('tab10', all_data.shape[0])  # Farbpalette (z.B. 'tab10' für 10 Farben)

    # Plotten der Trainingsdaten für jede Datei mit unterschiedlichen Farben
    for i in range(all_data.shape[0]):
        plt.scatter(X_test, all_data[i], color=colors(i), label=f'Trainingsdatei {i+1}', s=30)  # Jede Datei mit eigener Farbe
    
    # Plotten des Baseline-Modells (Mittelwert)
    plt.plot(X_test, baseline_means, 'g--', label='Baseline Mittelwert', linewidth=2)  # Baseline Modell
    
    plt.xlabel('x-Wert (Position)')
    plt.ylabel('y-Wert (Höhe)')
    plt.title('Gaussian Process Regression und Baseline Modell')
    plt.legend()
    plt.show()

# Anwendung des GPR auf Dateien und plotten
file_directory = r"B:\filtered_output_NaN_TH"
row_number = 100  # Beispiel: Reihe 100 in jeder Datei verwenden
X_train, y_train, X_test, y_pred, y_var, all_data = apply_gpr(file_directory, row_number)

# Berechnen des Baseline-Modells (Mittelwert für jede x-Position)
baseline_means = calculate_baseline_model(all_data)

# Plotten der Ergebnisse
plot_gpr_with_baseline(X_train, y_train, X_test, y_pred, y_var, baseline_means, all_data)
