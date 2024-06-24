import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GPy
from py_helpers import select_files
from tqdm import tqdm
from joblib import Parallel, delayed
import time

def train_gpr(X, y):
    kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)  # Feste Hyperparameter
    m = GPy.models.GPRegression(X, y, kernel)
    return m

def process_profile(df, profile_index):
    print(f"Processing profile {profile_index}")
    data_values = df.values[profile_index]
    x_values = np.arange(len(data_values)).reshape(-1, 1)
    gpr_model = train_gpr(x_values, data_values.reshape(-1, 1))
    y_pred, sigma = gpr_model.predict(x_values)
    return (x_values, data_values, y_pred, sigma)

def process_file(file):
    print(f"Processing file: {file}")
    df = pd.read_csv(file, header=None, delimiter=";")
    file_predictions = [process_profile(df, profile_index) for profile_index in range(df.shape[0])]
    return file, file_predictions

def main():
    file_paths = select_files()
    print(f"Selected files: {file_paths}")
    input_segments = {}
    predictions = {}

    for file in file_paths:
        input_segments[file] = pd.read_csv(file, header=None, delimiter=";")

    total_profiles_per_file = [df.shape[0] for df in input_segments.values()]
    min_profiles = min(total_profiles_per_file)
    max_profiles = max(total_profiles_per_file)

    print(f"Total profiles per file: {total_profiles_per_file}")
    print(f"Smallest number of profiles across all files: {min_profiles}")
    print(f"Biggest number of profiles across all files: {max_profiles}")

    total_steps = sum(total_profiles_per_file)
    progress_bar = tqdm(total=total_steps, desc="Training Progress", unit="profile")

    results = Parallel(n_jobs=-1)(delayed(process_file)(file) for file in file_paths)

    for file, file_predictions in results:
        predictions[file] = file_predictions
        progress_bar.update(len(file_predictions))

    progress_bar.close()

    print("Plotting results")
    plt.ion()  # Interactive mode on
    fig, ax = plt.subplots(figsize=(10, 6))

    for profile_index in range(max_profiles):
        ax.clear()
        ax.set_ylim(0, 1000)
        ax.set_title(f'Profile {profile_index+1}')
        ax.set_xlabel('Index der Messwerte')
        ax.set_ylabel('Höhenwerte')

        for file_index, file in enumerate(file_paths):
            if profile_index < len(predictions[file]):
                x_values, data_values, y_pred, sigma = predictions[file][profile_index]

                if file_index == 0:
                    ax.plot(x_values, data_values, 'k.', markersize=10, label='Original data')
                    ax.plot(x_values, y_pred, 'blue', label='Predicted mean')
                    ax.fill_between(x_values.flatten(),
                                    y_pred.flatten() - 1.96 * np.sqrt(sigma.flatten()),  # Konfidenzintervall basierend auf Standardabweichung
                                    y_pred.flatten() + 1.96 * np.sqrt(sigma.flatten()),
                                    alpha=0.5, color='red', label='95% confidence interval')
                else:
                    ax.plot(x_values, data_values, 'k.', markersize=10)
                    ax.plot(x_values, y_pred, 'blue')
                    ax.fill_between(x_values.flatten(),
                                    y_pred.flatten() - 1.96 * np.sqrt(sigma.flatten()),  # Konfidenzintervall basierend auf Standardabweichung
                                    y_pred.flatten() + 1.96 * np.sqrt(sigma.flatten()),
                                    alpha=0.5, color='red')

        ax.legend()
        plt.draw()
        plt.pause(2)  # Zeit in Sekunden, um jedes Diagramm anzuzeigen

    plt.ioff()  # Interactive mode off
    plt.show()

# run main
if __name__ == "__main__":
    main()
