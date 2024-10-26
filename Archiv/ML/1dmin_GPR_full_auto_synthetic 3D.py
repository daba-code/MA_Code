import numpy as np
import GPy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from scipy.stats import ttest_rel

# Funktion zur Berechnung des RMSE
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Funktion zur Berechnung des MAE
def mean_absolute_error_custom(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Funktion zur Berechnung des R^2 Werts
def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

# Schritt 1: Erzeugung von simulierten 3D-Daten mit Rauschen
np.random.seed(42)
num_files = 3       # Anzahl der simulierten Dateien
num_profiles = 20   # Anzahl der Profile in jeder Datei
num_columns = 100   # Anzahl der Datenpunkte pro Profil

# Generierung eines komplexen 3D-Datensatzes
x_values = np.linspace(0, 4 * np.pi, num_columns)
all_measurements = np.array([
    np.vstack([
        np.sin(x_values * (1 + 0.1 * row)) +
        0.5 * np.sin(x_values * (1 + 0.05 * row + 1)) +
        0.3 * np.cos(x_values * (1 + 0.03 * row + 2)) +
        np.random.normal(0, 0.2, num_columns)
        for row in range(num_profiles)
    ]) for _ in range(num_files)
])  # Shape: (num_files, num_profiles, num_columns)

# K-Fold Cross-Validation Setup für Dateien
k = 2  # Anzahl der Folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Speichern der p-Werte und Metriken für jede Datei
file_metrics = {}
file_p_values = {}

# Durchlauf durch jede Datei und Anwendung von K-Fold Cross-Validation
for file_index, (train_indices, val_indices) in enumerate(kf.split(all_measurements)):
    print(f"\nK-Fold Cross-Validation für Datei {file_index + 1} mit {k}-Folds")

    # Training- und Validierungs-Datensätze
    train_data = all_measurements[train_indices]
    val_data = all_measurements[val_indices]
    
    # Baseline und GPR Residuen-Listen initialisieren
    residuals_gpr = []
    residuals_baseline = []
    rmse_gpr_list, mae_gpr_list, r2_gpr_list = [], [], []
    rmse_baseline_list, mae_baseline_list, r2_baseline_list = [], [], []

    # Baseline: Mittelwert über alle Trainingsdateien für jeden Punkt
    baseline_pred = train_data.mean(axis=0)

    # 2D-GPR-Modell vorbereiten
    X_2d = np.array([[x, y] for y in range(num_profiles) for x in range(num_columns)])
    Y_train = baseline_pred.flatten().reshape(-1, 1)

    # GPR-Modell trainieren
    kernel_2d = GPy.kern.RBF(input_dim=2, variance=0.5, lengthscale=2)
    model_2d = GPy.models.GPRegression(X_2d, Y_train, kernel_2d)
    model_2d.optimize(messages=False)

    # Plot für jedes Validierungsprofil
    for val_file_index, val_file in enumerate(val_data):
        Y_true = val_file.flatten().reshape(-1, 1)
        Y_pred_2d, Y_pred_var_2d = model_2d.predict(X_2d)
        Y_pred_std_2d = np.sqrt(Y_pred_var_2d).flatten()

        # Berechnung der Metriken
        rmse_baseline = root_mean_squared_error(Y_true, baseline_pred.flatten())
        mae_baseline = mean_absolute_error_custom(Y_true, baseline_pred.flatten())
        r2_baseline_value = r2(Y_true, baseline_pred.flatten())

        rmse_gpr = root_mean_squared_error(Y_true, Y_pred_2d)
        mae_gpr = mean_absolute_error_custom(Y_true, Y_pred_2d)
        r2_gpr_value = r2(Y_true, Y_pred_2d)

        # Residuen berechnen für t-Test
        residual_baseline = Y_true.flatten() - baseline_pred.flatten()
        residual_gpr = Y_true.flatten() - Y_pred_2d.flatten()
        residuals_baseline.extend(residual_baseline)
        residuals_gpr.extend(residual_gpr)

        # Plot für das aktuelle Profil
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(f"Profil aus Datei {file_index + 1}, Validierungsprofil {val_file_index + 1}")
        
        for profile_index in range(num_profiles):
            # Plot der tatsächlichen Werte
            ax.plot(x_values, val_file[profile_index], color="blue", label="Tatsächliche Werte" if profile_index == 0 else "")
            # Plot der Baseline-Vorhersage
            ax.plot(x_values, baseline_pred[profile_index], color="green", linestyle="--", label="Baseline" if profile_index == 0 else "")
            # Plot der GPR-Vorhersage und des Konfidenzintervalls
            Y_pred_profile = Y_pred_2d[profile_index*num_columns:(profile_index+1)*num_columns].flatten()
            Y_pred_std_profile = Y_pred_std_2d[profile_index*num_columns:(profile_index+1)*num_columns]
            ax.plot(x_values, Y_pred_profile, color="red", label="GPR-Vorhersage" if profile_index == 0 else "")
            ax.fill_between(x_values, Y_pred_profile - 1.96 * Y_pred_std_profile, Y_pred_profile + 1.96 * Y_pred_std_profile, color="orange", alpha=0.5, label="95% Konfidenzintervall" if profile_index == 0 else "")

        ax.set_xlabel("Position entlang des Profils")
        ax.set_ylabel("Höhe")
        ax.legend()
        plt.show()

    # Gepaarten t-Test für die Residuen der aktuellen Datei durchführen
    t_stat, p_value = ttest_rel(residuals_baseline, residuals_gpr)
    file_p_values[f"Datei {file_index + 1}"] = p_value

    print(f"\nDatei {file_index + 1} - p-Wert für den Vergleich der Residuen: {p_value:.4f}")
    if p_value < 0.05:
        print("Ergebnis: Die Verbesserung des GPR-Modells gegenüber dem Baseline-Modell ist für diese Datei statistisch signifikant.")
    else:
        print("Ergebnis: Keine signifikante Verbesserung des GPR-Modells gegenüber dem Baseline-Modell für diese Datei.")
