import numpy as np
import matplotlib.pyplot as plt
import GPy
import pandas as pd
import glob
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from scipy.stats import ttest_rel


# Funktion zur Berechnung des RMSE
def root_mean_squared_error(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

# Funktion zur Berechnung des MAE
def mean_absolute_error_custom(y_test, y_pred):
    return mean_absolute_error(y_test, y_pred)

# Funktion zur Berechnung des R^2 Werts
def r2(y_test, y_pred):
    return r2_score(y_test, y_pred)

def read_row_from_csv(file_path, row_number):
    """
    Liest eine bestimmte Zeile aus einer CSV-Datei ein.
    
    :param file_path: Pfad zur CSV-Datei
    :param row_number: Nummer der Zeile, die eingelesen werden soll (beginnend bei 0)
    :return: Array mit den Werten der Zeile
    """
    # CSV-Datei einlesen
    df = pd.read_csv(file_path, delimiter=';')
    
    # Die gewünschte Zeile extrahieren (row_number gibt die gewünschte Zeile an)
    y_train = df.iloc[row_number].values  # Gibt die Werte der gewünschten Zeile zurück
    
    if len(y_train) == 383:
            return y_train
    else:
            print(f"Warnung: Datei {file_path} hat weniger als 383 Werte in Zeile {row_number}")
            return None

def load_y_train_from_directory(directory, row_number):
    """
    Liest die angegebene Reihe aus allen CSV-Dateien im Verzeichnis und speichert sie als `y_train_n` Arrays.
    
    :param directory: Verzeichnis, in dem sich die CSV-Dateien befinden
    :param row_number: Die Zeilennummer, die aus jeder CSV-Datei geladen werden soll
    :return: Liste von Arrays, jedes Array entspricht einer Zeile aus einer CSV-Datei
    """
    # Liste der CSV-Dateien im Verzeichnis abrufen
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    # Liste zur Speicherung der y_train Arrays
    y_train_list = []

    # Jede CSV-Datei durchlaufen und die gewünschte Reihe einlesen
    for file in csv_files:
        y_train = read_row_from_csv(file, row_number)
        y_train_list.append(y_train)
    
    return y_train_list  # Gibt eine Liste von y_train Arrays zurück

directory = r"B:\filtered_output\gekuerzt\aligned_files\processed_files\gekuerzt"
row_number = 100

# 1. Erstellen der Trainingsdaten von 0 bis 383
X_train = np.linspace(0, 383, 383).reshape(-1, 1)  # 383 x-Werte von 0 bis 383
y_train_list = load_y_train_from_directory(directory, row_number)

X_train_combined = np.tile(X_train, (len(y_train_list), 1))  # Dupliziere die x-Werte für beide y-Arrays
y_train_combined = np.hstack(y_train_list) # Kombiniere beide y-Werte in einem Array

# 2. Definieren des Kernels in GPy
kernel = GPy.kern.RatQuad(input_dim=1, variance=650, lengthscale=100, power=0.6)

# Setze Grenzen (Bounds) für die Parameter:
#kernel.variance.constrain_bounded(0.1, 100)  # Variance zwischen 0.1 und 100
#kernel.lengthscale.constrain_bounded(0.1, 10)  # Lengthscale zwischen 0.1 und 50
#kernel.power.constrain_bounded(0.01, 2)  # Power zwischen 0.01 und 2

# 3. GPR-Modell erstellen und trainieren
model = GPy.models.GPRegression(X_train_combined, y_train_combined.reshape(-1, 1), kernel)
#model.Gaussian_noise.variance.constrain_bounded(0.1, 1000)

# 4. Optimierung des Modells
model.optimize(messages=True, max_iters=1000)

# Optimierte Parameter ausgeben
print("Optimierte Kernel-Parameter:")
print(f"Variance: {model.kern.variance.values[0]}")
print(f"Lengthscale: {model.kern.lengthscale.values[0]}")
print(f"Power: {model.kern.power.values[0]}")

print("Optimierte Rauschvarianz:")
print(f"Noise variance: {model.Gaussian_noise.variance.values[0]}")

# 5. Vorhersagen für die Trainingsdaten
y_pred_train, y_var_train = model.predict(X_train)
y_std_train = np.sqrt(y_var_train)  # Standardabweichung für Konfidenzintervall der Trainingsdaten

#Baseline Modell trainieren
y_baseline = np.mean(y_train_list, axis = 0)

#vorhersage für die Testdaten
#X_test = X_train
#y_pred_test, y_var_test = model.predict(X_test)
#y_std_test = np.sqrt(y_var_test)

# 6. Plotten der Ergebnisse
plt.figure(figsize=(12, 6))

# Annahme: y_train_list enthält alle y_train Arrays, die fürs Training verwendet werden
for idx, y_train in enumerate(y_train_list):
    plt.scatter(X_train, y_train, alpha=0.3, label=f'Trainingsdaten {idx + 1}')  # Plotte jedes Profil aus y_train_list


# GPR Vorhersage für die Trainingsdaten und Unsicherheitsbereich
plt.plot(X_train, y_pred_train, 'k:', label='GPy Vorhersage', linewidth = 4)
plt.fill_between(
    X_train.ravel(),
    y_pred_train.flatten() - 1.96 * y_std_train.flatten(),
    y_pred_train.flatten() + 1.96 * y_std_train.flatten(),
    alpha=0.2,
    color='gray',
    label='95% Konfidenzintervall'
)
# plot des Baseline Modells
plt.plot(X_train, y_baseline, 'g--', label='Baseline (Mittelwert)', linewidth=2)


"""
# Plot der Testdaten
plt.scatter(X_train, y_test, color='red', label='Testdaten')

# GPR Vorhersage für die Testdaten und Unsicherheitsbereich
plt.plot(X_train, y_pred_test, 'b--', label='GPy Vorhersage (Testdaten)')
plt.fill_between(
    X_train.ravel(),
    y_pred_test.flatten() - 1.96 * y_std_test.flatten(),
    y_pred_test.flatten() + 1.96 * y_std_test.flatten(),
    alpha=0.2,
    color='blue',
    label='95% Konfidenzintervall (Testdaten)'
)
"""
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.title("GPy-GPR Vorhersage für Trainingsdaten und Testdaten")
plt.show()
