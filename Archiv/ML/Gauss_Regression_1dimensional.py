import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GPy
from py_helpers import select_files
from tqdm import tqdm
from joblib import Parallel, delayed

def train_gpr(X, y):
    print(f"Training GPR model with {X.shape[0]} data points")
    kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)  # Fixed hyperparameters
    m = GPy.models.GPRegression(X, y, kernel)
    m.optimize(messages=True)  # Optional: Optimize the model hyperparameters
    return m

def process_profile_combined(df_list, profile_index):
    print(f"Processing combined profile {profile_index}")
    combined_data = []
    combined_indices = []
    
    for df in df_list:
        if profile_index < df.shape[0]:
            data_values = df.values[profile_index]
            indices = np.arange(len(data_values))
            combined_data.append(data_values)
            combined_indices.append(indices)

    # Flatten combined data and indices, intersperse with NaNs to avoid connecting lines
    flat_data = np.concatenate([np.append(arr, [np.nan]) for arr in combined_data])
    flat_indices = np.concatenate([np.append(arr, [np.nan]) for arr in combined_indices])

    x_values = flat_indices.reshape(-1, 1)
    y_values = flat_data.reshape(-1, 1)
    
    valid_mask = ~np.isnan(x_values.flatten())
    gpr_model = train_gpr(x_values[valid_mask].reshape(-1, 1), y_values[valid_mask].reshape(-1, 1))
    y_pred, sigma = gpr_model.predict(x_values[valid_mask].reshape(-1, 1))
    
    # Re-insert NaNs to the predicted values and sigma
    y_pred_full = np.full_like(x_values, np.nan, dtype=np.float64)
    sigma_full = np.full_like(x_values, np.nan, dtype=np.float64)
    y_pred_full[valid_mask] = y_pred
    sigma_full[valid_mask] = sigma
    
    return profile_index, (x_values, y_values, y_pred_full, sigma_full)

def process_files(file_paths):
    print(f"Reading data from files: {file_paths}")
    dataframes = [pd.read_csv(file, header=None, delimiter=";") for file in file_paths]
    max_profiles = max(df.shape[0] for df in dataframes)
    
    file_predictions = Parallel(n_jobs=-1)(
        delayed(process_profile_combined)(dataframes, profile_index)
        for profile_index in range(max_profiles)
    )
    
    return sorted(file_predictions, key=lambda x: x[0])  # Sort by profile_index

def main():
    file_paths = select_files()
    print(f"Selected files: {file_paths}")
    
    predictions = process_files(file_paths)
    max_profiles = len(predictions)

    print("Plotting results")
    plt.ion()  # Interactive mode on
    fig, ax = plt.subplots(figsize=(10, 6))

    for profile_index, prediction in predictions:
        ax.clear()  # Clear the current axes.
        ax.set_ylim(0, 1000)
        ax.set_title(f'Profile {profile_index+1}')
        ax.set_xlabel('Index der Messwerte')
        ax.set_ylabel('Höhenwerte')

        x_values, data_values, y_pred, sigma = prediction

        # Plot original data
        ax.plot(x_values, data_values, linestyle='-', marker='.', color='k', markersize=4, linewidth = 2, label='Original data', alpha=0.5)
        # Plot predicted mean
        ax.plot(x_values, y_pred, 'blue', label='Predicted mean')
        # Plot confidence interval
        ax.fill_between(x_values.flatten(),
                        y_pred.flatten() - 1.96 * np.sqrt(sigma.flatten()),  # Confidence interval based on standard deviation
                        y_pred.flatten() + 1.96 * np.sqrt(sigma.flatten()),
                        alpha=0.5, color='red', label='95% confidence interval')

        ax.legend()
        plt.draw()
        plt.pause(2)  # Time in seconds to display each plot
        print(f"Plotted profile {profile_index+1}")

    plt.ioff()  # Interactive mode off
    plt.show()
    print("Plotting completed")

# Run main
if __name__ == "__main__":
    main()
