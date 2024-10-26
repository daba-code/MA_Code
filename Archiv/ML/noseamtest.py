import pandas as pd
import numpy as np
import GPy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Funktion zur Berechnung des RMSE
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Funktion zur Berechnung des R^2 Werts
def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

# Schritt 1: Erzeugung von simulierten Sinus-Daten mit Rauschen
np.random.seed(42)
num_files = 10  # Anzahl der simulierten Dateien
num_columns = 350  # Anzahl der Spalten in jeder Datei (entspricht der Anzahl der Punkte in der Sinuskurve)

sin_noise = 0.1  # Rauschgröße für den Sinus
cos_noise = 0.5  # Rauschgröße für den Cosinus

# Erstellen von sinus- und cosinusförmigen Daten mit Rauschen für mehrere Zeilen (z. B. 10 Dateien, jede mit 2 Zeilen)
x_values = np.linspace(0, 4 * np.pi, num_columns)
all_measurements = [
    pd.DataFrame({
        "sin": np.sin(x_values) + np.random.normal(0, sin_noise, num_columns),
        "cos": np.cos(x_values) + np.random.normal(0, cos_noise, num_columns)
    }).T for _ in range(num_files)
]

# Schritt 2: Aufteilung der Daten in Trainings- (80%) und Validierungsdateien (20%)
train_size = int(0.8 * len(all_measurements))  # 80% der Daten für das Training
train_files = all_measurements[:train_size]  # Ersten 80% der Dateien als Trainingsdaten
val_files = all_measurements[train_size:]    # Letzten 20% der Dateien als Validierungsdaten

# Schritt 3: Initialisierung des Plots
plt.ion()  # Interaktiver Modus aktivieren
fig, ax = plt.subplots(figsize=(12, 6))

# Durchlauf durch jede Zeile, um das GPR-Modell mit den Trainingsdaten zu trainieren und auf den Validierungsdaten zu testen
num_rows = train_files[0].shape[0]  # Anzahl der Zeilen in jeder Datei

for row_index in range(num_rows):
    print(f"\nTrainiere GPR-Modell für Zeile {row_index + 1} mit 80% Training und 20% Validierung")

    # Sammeln der Daten für die aktuelle Zeile aus den Trainingsdateien
    row_data_train = [df.iloc[row_index].values for df in train_files if len(df.iloc[row_index].dropna()) == num_columns]
    if len(row_data_train) == 0:
        print(f"Zeile {row_index + 1} übersprungen - unvollständige Daten")
        continue
        
    row_data_train = np.array(row_data_train)  # Form: (Anzahl Trainingsdateien, 150)

    # Vorbereitung von X und Y für das GPR-Modell für diese Zeile
    X = np.arange(num_columns).reshape(-1, 1)  # Positionen entlang des Profils (0 bis 149)
    Y_train = row_data_train.mean(axis=0).reshape(-1, 1)  # Durchschnittsprofil über die Trainingsdateien

    # Sicherstellen, dass X und Y_train die gleiche Anzahl an Zeilen haben
    if X.shape[0] != Y_train.shape[0]:
        print("Fehler: Die Anzahl der Datenpunkte in X und Y_train stimmt nicht überein.")
        continue

    # Definition eines GPR-Modells für die aktuelle Zeile mit einem Radial Basis Function (RBF)-Kernel
    kernel = GPy.kern.RBF(input_dim=1, variance=0.5, lengthscale=2)
    model = GPy.models.GPRegression(X, Y_train, kernel)

    # Modelloptimierung für diese Zeile mit den Trainingsdaten
    model.optimize(messages=True)

    # Validierung auf den letzten 20% Dateien und Berechnung des RMSE und R^2
    rmse_scores = []
    r2_scores = []
    for df_val in val_files:
        val_data = df_val.iloc[row_index].values.reshape(-1, 1)  # Aktuelle Zeilendaten der Validierungsdatei
        if val_data.shape[0] != num_columns:
            print(f"Validierungsdatei für Zeile {row_index + 1} übersprungen - unvollständige Daten")
            continue
            
        Y_pred, _ = model.predict(X)
        
        # Berechnung von RMSE und R^2
        rmse = root_mean_squared_error(val_data, Y_pred)
        r2_value = r2(val_data, Y_pred)
        
        rmse_scores.append(rmse)
        r2_scores.append(r2_value)

    # Berechnung des durchschnittlichen RMSE und R^2 für die aktuelle Zeile über die Validierungsdateien
    if rmse_scores:
        avg_rmse = np.mean(rmse_scores)
        avg_r2 = np.mean(r2_scores)
        print(f"Zeile {row_index + 1} - Durchschnittlicher RMSE auf Validierungsdateien: {avg_rmse:.4f}")
        print(f"Zeile {row_index + 1} - Durchschnittlicher R^2 auf Validierungsdateien: {avg_r2:.4f}")

    # Plotten des GPR-Modells basierend auf den Trainingsdaten für die aktuelle Zeile
    Y_pred_full, Y_pred_var_full = model.predict(X)
    Y_pred_std_full = np.sqrt(Y_pred_var_full)

    # Aktualisierung des Plots mit dem neuen Profil
    ax.clear()
    ax.plot(X, Y_pred_full, color='red', label='GPR Mittelwert')
    ax.fill_between(
        X.flatten(),
        (Y_pred_full - 1.96 * Y_pred_std_full).flatten(),
        (Y_pred_full + 1.96 * Y_pred_std_full).flatten(),
        color='orange',
        alpha=0.2,
        label='95% Konfidenzintervall'
    )

    # Hinzufügen der Datenpunkte der Trainingsdateien für die aktuelle Zeile
    for i, train_data in enumerate(row_data_train):
        ax.scatter(X, train_data, label=f'Trainingsdatei {i + 1}', alpha=0.6, s=10)  # Datenpunkte für jede Datei

    ax.set_title(f"GPR Modell für Zeile {row_index + 1} (80% Training, 20% Validierung)")
    ax.set_xlabel("Position entlang des Profils")
    ax.set_ylabel("Erwartungswert")
    ax.legend()

    plt.pause(5)  # Pause, um den Plot zu aktualisieren; Pausenzeit nach Bedarf anpassen

# Interaktiven Modus deaktivieren und den finalen Plot anzeigen
plt.ioff()
plt.show()
