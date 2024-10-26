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
        df = pd.read_csv(file_path, sep=";", header=None)
        if row_number < len(df):
            row_data = df.iloc[row_number].values
            all_data.append(row_data)
    
    return np.array(all_data)

# Funktion zum Berechnen des Baseline-Modells (Mittelwert der y-Werte für jeden x-Wert)
def calculate_baseline_model(all_data):
    baseline_means = np.nanmean(all_data, axis=0)
    return baseline_means

# Funktion zum Filtern und Entfernen von NaNs
def preprocess_data_for_gpr(x_positions, all_data):
    X_train_combined = []
    y_train_combined = []
    for i in range(all_data.shape[0]):
        y_values = all_data[i, :]
        valid_indices = ~np.isnan(y_values)
        X_train_combined.append(x_positions[valid_indices].reshape(-1, 1))
        y_train_combined.append(y_values[valid_indices].reshape(-1, 1))
    X_train_combined = np.vstack(X_train_combined)
    y_train_combined = np.vstack(y_train_combined)
    return X_train_combined, y_train_combined

# Beispiel: Verwenden des GPR-Modells auf den geladenen Daten
def apply_gpr(file_directory, row_number=100):
    all_data = load_row_from_files(file_directory, row_number=row_number)
    x_positions = np.arange(383)
    X_train, y_train = preprocess_data_for_gpr(x_positions, all_data)
    
    kernel = GPy.kern.Matern32(input_dim=1, variance=1, lengthscale=1, ARD=True)
    kernel.variance.constrain_bounded(0.1, 1100)
    kernel.lengthscale.constrain_bounded(0.1, 60)
    
    model = GPy.models.GPRegression(X_train, y_train, kernel)
    model.optimize(messages=True)
    
    X_test = x_positions.reshape(-1, 1)
    y_pred, y_var = model.predict(X_test)
    print(model)
    return X_train, y_train, X_test, y_pred, y_var, all_data

# Funktion zur Analyse eines größeren Fensters um die größte negative Steigung
def find_max_negative_slope_window(gpr_profile, window_size=10):
    derivative = np.gradient(gpr_profile)
    derivative[:10] = np.nan
    derivative[-10:] = np.nan
    negative_derivative = np.where((derivative < 0), derivative, np.nan)
    
    if np.isnan(negative_derivative).all():
        return None, None

    max_negative_slope_index = np.nanargmin(negative_derivative)
    
    if max_negative_slope_index is not None and max_negative_slope_index > window_size:
        if gpr_profile[max_negative_slope_index - 1] > gpr_profile[max_negative_slope_index]:
            start = max(max_negative_slope_index - window_size // 2, 0)
            end = min(max_negative_slope_index + window_size // 2 + 1, len(gpr_profile))
            avg_slope_position = np.nanmean(np.arange(start, end))
            return max_negative_slope_index, int(avg_slope_position)
    return None, None

# Plotten der Vorhersagen, Konfidenzintervall, Trainingsdaten und des Baseline-Modells
def plot_gpr_with_baseline_and_derivative(X_train, y_train, X_test, y_pred, y_var, baseline_means, all_data):
    y_std = np.sqrt(y_var)
    lower_bound = y_pred - 1.96 * y_std
    upper_bound = y_pred + 1.96 * y_std

    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_pred, 'r-', label='GPR Vorhersage')
    plt.fill_between(X_test.flatten(), lower_bound.flatten(), upper_bound.flatten(), 
                     alpha=0.2, color='gray', label='95% Konfidenzintervall')
    
    colors = plt.cm.get_cmap('tab10', all_data.shape[0])
    for i in range(all_data.shape[0]):
        plt.scatter(X_test, all_data[i], color=colors(i), label=f'Trainingsdatei {i+1}', s=30)
    
    plt.plot(X_test, baseline_means, 'g--', label='Baseline Mittelwert', linewidth=2)
    plt.xlabel('x-Wert (Position)')
    plt.ylabel('y-Wert (Höhe)')
    plt.title('Gaussian Process Regression und Baseline Modell')
    plt.legend()
    plt.show()

    # Ableitung berechnen und Plotten
    derivative = np.gradient(y_pred.flatten())
    derivative[:10] = np.nan
    derivative[-10:] = np.nan
    second_derivative = np.gradient(derivative)

    plt.figure(figsize=(10, 4))
    plt.plot(X_test.flatten(), derivative, 'b-', label='Erste Ableitung')
    plt.plot(X_test.flatten(), second_derivative, 'g--', label='Zweite Ableitung')
    
    max_neg_slope_pos, avg_slope_position = find_max_negative_slope_window(y_pred.flatten(), window_size=10)
    if max_neg_slope_pos is not None:
        plt.plot(X_test[max_neg_slope_pos], derivative[max_neg_slope_pos], 'ro', label=f'Größte neg. Steigung (Index {max_neg_slope_pos})')
        plt.axvline(x=X_test[int(avg_slope_position)], color='purple', linestyle='--', label=f'Fenstermittelpunkt (Index {avg_slope_position})')

    plt.xlabel('x-Wert (Position)')
    plt.ylabel('Ableitung')
    plt.title('Erste und Zweite Ableitung der GPR-Vorhersage')
    plt.legend()
    plt.show()

# Anwendung des GPR auf Dateien und plotten
file_directory = r"B:\filtered_output_NaN_TH"
row_number = 100
X_train, y_train, X_test, y_pred, y_var, all_data = apply_gpr(file_directory, row_number)

baseline_means = calculate_baseline_model(all_data)
plot_gpr_with_baseline_and_derivative(X_train, y_train, X_test, y_pred, y_var, baseline_means, all_data)
