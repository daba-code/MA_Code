import pandas as pd
import numpy as np
import GPy
import glob
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Funktion zur Berechnung des RMSE, MAE und R², wobei NaNs herausgefiltert werden
def calculate_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    rmse = int(round(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))))
    mae = int(round(mean_absolute_error(y_true[mask], y_pred[mask])))
    r2 = round(r2_score(y_true[mask], y_pred[mask]), 4)
    return rmse, mae, r2

# Schritt 1: Laden der Daten aus allen Dateien in eine Liste von DataFrames
file_paths = glob.glob(r"B:\temp\*.csv")  # Pfad nach Bedarf anpassen
all_measurements = []
nan_info = {}  # Dictionary zum Speichern der NaN-Informationen pro Zeile

# Speicherort für die JSON-Datei festlegen
json_file_path = r"B:\nan_info.json"

if not file_paths:
    print("Keine CSV-Dateien im angegebenen Verzeichnis gefunden.")
else:
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=";", dtype=int).replace(0, np.nan)
        if df.shape[1] == 150:
            # Nullwerte durch NaN ersetzen und zur Liste hinzufügen
            df = df.apply(pd.to_numeric, errors='coerce').replace(0, np.nan)
            all_measurements.append(df)
            
            # NaN-Informationen sammeln
            for index, row in df.iterrows():
                nan_columns = row.index[row.isna()].tolist()  # Liste der Spalten mit NaN
                if index not in nan_info:
                    nan_info[index] = nan_columns
                else:
                    nan_info[index].extend(nan_columns)
        else:
            print(f"Datei {file_path} hat nicht 150 Spalten. Wird übersprungen.")

# Speichern der NaN-Informationen in einer JSON-Datei am gewünschten Speicherort
with open(json_file_path, "w") as f:
    json.dump(nan_info, f, indent=4)

# Schritt 2: Datenaufteilung in Trainings- (80%) und Validierungsdateien (20%)
num_files = len(all_measurements)
train_size = int(0.8 * num_files)
train_files = all_measurements[:train_size]
val_files = all_measurements[train_size:]

# Schritt 3: Initialisierung des Plots
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))

# Durchlauf durch jede Zeile für das GPR-Modelltraining und Validierung
num_rows = train_files[0].shape[0]

for row_index in range(num_rows):
    print(f"\nTrainiere GPR-Modell für Zeile {row_index + 1} mit 80% Training und 20% Validierung")

    row_data_train = [df.iloc[row_index].values for df in train_files]
    row_data_train = np.array(row_data_train)

    valid_indices = ~np.isnan(row_data_train.mean(axis=0))
    X_valid = np.arange(150)[valid_indices].reshape(-1, 1)
    Y_train = np.round(row_data_train[:, valid_indices].mean(axis=0)).reshape(-1, 1).astype(int)

    scaler = StandardScaler()
    Y_train_scaled = scaler.fit_transform(Y_train)

    kernel = GPy.kern.Matern52(input_dim=1, variance=5.0, lengthscale=1.0) + GPy.kern.Linear(input_dim=1)
    model = GPy.models.GPRegression(X_valid, Y_train_scaled, kernel)

    model.Gaussian_noise.variance = 5.0
    model.optimize(messages=True)

    rmse_scores = []
    mae_scores = []
    r2_scores = []

    for df_val in val_files:
        val_data = df_val.iloc[row_index].values.reshape(-1, 1)
        val_data = val_data[valid_indices]

        val_data_scaled = scaler.transform(val_data)
        Y_pred_scaled, _ = model.predict(X_valid)
        Y_pred = scaler.inverse_transform(Y_pred_scaled)
        Y_pred = np.round(Y_pred).astype(int)  # Rundung auf Ganzzahl

        rmse, mae, r2 = calculate_metrics(val_data, Y_pred)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)

    avg_rmse = int(round(np.mean(rmse_scores)))
    avg_mae = int(round(np.mean(mae_scores)))
    avg_r2 = round(np.mean(r2_scores), 4)
    print(f"Zeile {row_index + 1} - Durchschnittlicher RMSE: {avg_rmse}, MAE: {avg_mae}, R²: {avg_r2}")

    X_full = np.arange(150).reshape(-1, 1)
    Y_pred_full_scaled, Y_pred_var_full = model.predict(X_full)
    Y_pred_full = scaler.inverse_transform(Y_pred_full_scaled)
    Y_pred_full = np.round(Y_pred_full).astype(int)  # Rundung auf Ganzzahl
    Y_pred_std_full = np.sqrt(Y_pred_var_full)

    ax.clear()
    ax.plot(X_full, Y_pred_full, color='red', label='GPR Mittelwert')
    ax.fill_between(
        X_full.flatten(),
        (Y_pred_full - 1.96 * Y_pred_std_full).flatten(),
        (Y_pred_full + 1.96 * Y_pred_std_full).flatten(),
        color='orange',
        alpha=0.2,
        label='95% Konfidenzintervall'
    )

    for i, train_data in enumerate(row_data_train):
        ax.scatter(X_valid, train_data[valid_indices], label=f'Trainingsdatei {i + 1}', alpha=0.6, s=10)

    ax.set_title(f"GPR Modell für Zeile {row_index + 1} (80% Training, 20% Validierung)")
    ax.set_xlabel("Position entlang des Profils")
    ax.set_ylabel("Erwartungswert")
    ax.legend()

    plt.pause(0.5)

plt.ioff()
plt.show()
print(f"NaN-Informationen gespeichert in '{json_file_path}'")
