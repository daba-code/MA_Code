import pandas as pd 
import numpy as np
import GPy
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Funktion zur Berechnung des RMSE
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Schritt 1: Laden der Daten aus allen Dateien in eine Liste von DataFrames
file_paths = glob.glob(r"B:\temp\*.csv")  # Pfad nach Bedarf anpassen
all_measurements = []

if not file_paths:
    print("Keine CSV-Dateien im angegebenen Verzeichnis gefunden.")
else:
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=";")
        if df.shape[1] == 150:
            df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any')
            if not df.isna().any().any():
                all_measurements.append(df)  # Hinzufügen des DataFrames jeder Datei
            else:
                print(f"Datei {file_path} enthält nach der Umwandlung NaNs. Wird übersprungen.")
        else:
            print(f"Datei {file_path} hat nicht 150 Spalten. Wird übersprungen.")

# Schritt 2: Datenaufteilung in Trainings- (80%) und Validierungsdateien (20%)
num_files = len(all_measurements)
train_size = int(0.8 * num_files)  # 80% der Daten für das Training
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
    row_data_train = [df.iloc[row_index].values for df in train_files]
    row_data_train = np.array(row_data_train)  # Form: (80% der Dateien, 150)

    # Vorbereitung von X und Y für das GPR-Modell für diese Zeile
    X = np.arange(150).reshape(-1, 1)  # Positionen entlang des Profils (0 bis 149)
    Y_train = row_data_train.mean(axis=0).reshape(-1, 1)  # Durchschnittsprofil über die Trainingsdateien

    # Definition eines GPR-Modells für die aktuelle Zeile mit einem Radial Basis Function (RBF)-Kernel
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=10.)
    model = GPy.models.GPRegression(X, Y_train, kernel)

    # Modelloptimierung für diese Zeile mit den Trainingsdaten
    model.optimize(messages=False)

    # Validierung auf den letzten 20% Dateien und Berechnung des RMSE
    rmse_scores = []
    for df_val in val_files:
        val_data = df_val.iloc[row_index].values.reshape(-1, 1)  # Aktuelle Zeilendaten der Validierungsdatei
        Y_pred, _ = model.predict(X)
        rmse = root_mean_squared_error(val_data, Y_pred)
        rmse_scores.append(rmse)

    # Berechnung des durchschnittlichen RMSE für die aktuelle Zeile über die Validierungsdateien
    avg_rmse = np.mean(rmse_scores)
    print(f"Zeile {row_index + 1} - Durchschnittlicher RMSE auf Validierungsdateien: {avg_rmse:.4f}")

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

    plt.pause(0.5)  # Pause, um den Plot zu aktualisieren; Pausenzeit nach Bedarf anpassen

# Interaktiven Modus deaktivieren und den finalen Plot anzeigen
plt.ioff()
plt.show()
