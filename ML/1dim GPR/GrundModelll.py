import numpy as np
import matplotlib.pyplot as plt
import GPy
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import ttest_rel, mannwhitneyu
from scipy.interpolate import interp1d
import pandas as pd  # Import pandas to handle data grouping

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate Training Data with Increased Number of Points (No x-shifts)
X_train = np.linspace(0, 10, 50).reshape(-1, 1)

# Define different noise levels
noise_level_1 = 0.2
noise_level_2 = 0.4
noise_level_3 = 0.7

# Generate y-values for three sine curves with different noise levels
y_train_1_noise1 = np.sin(X_train) + np.random.normal(0, noise_level_1, X_train.shape)
y_train_2_noise1 = np.sin(X_train) + np.random.normal(0, noise_level_2, X_train.shape)
y_train_3_noise1 = np.sin(X_train) + np.random.normal(0, noise_level_3, X_train.shape)

# 2. Combine Training Data (No x-shifts applied)
X_train_combined = np.vstack([
    X_train, X_train, X_train
]).flatten()

y_train_combined = np.hstack([
    y_train_1_noise1.flatten(), y_train_2_noise1.flatten(), y_train_3_noise1.flatten()
])

# Sort the training data by x-values
sorted_indices = np.argsort(X_train_combined)
X_train_combined = X_train_combined[sorted_indices]
y_train_combined = y_train_combined[sorted_indices]

# 3. Define the Kernel for GPR
kernel = GPy.kern.Matern32(input_dim=1, variance=1.0, lengthscale=1.0)

#kernel.lengthscale.constrain_bounded(0.1, 0.5)  # Lengthscale zwischen 0.1 und 50

# 4. Create and Train the GPR Model
model = GPy.models.GPRegression(X_train_combined.reshape(-1, 1), y_train_combined.reshape(-1, 1), kernel)
model.Gaussian_noise.variance = 0.1  # Set initial noise variance

# Optimize the model
model.optimize(messages=True, max_iters=1000)

print("Optimized Kernel Parameters:")
print(f"Variance: {model.kern.variance.values[0]:.4f}")
print(f"Lengthscale: {model.kern.lengthscale.values[0]:.4f}")

print("Optimized Noise Variance:")
print(f"Noise variance: {model.Gaussian_noise.variance.values[0]:.4f}")

# 5. Generate Test Data with Same Number of Points (No x-shifts applied)
X_test = np.linspace(0, 10, 50).reshape(-1, 1)
y_test = np.sin(X_test) + np.random.normal(0, noise_level_1, X_test.shape)

# Create the test dataset without anomalies
y_test_no_anomaly = y_test.copy()

# Create the test dataset with anomalies
anomaly_indices = (X_test.flatten() > 4) & (X_test.flatten() < 6)
y_test_anomalous = y_test.copy()
y_test_anomalous[anomaly_indices] += 1.5  # Introduce anomalies

# 6. Baseline Model Prediction for Both Test Datasets

# Create a DataFrame from the training data
df_train = pd.DataFrame({'X': X_train_combined, 'y': y_train_combined})

# Group by x-values and compute the mean y-value for each x
df_train_mean = df_train.groupby('X').mean().reset_index()

# Ensure that x-values are unique and sorted
X_unique = df_train_mean['X'].values
y_mean = df_train_mean['y'].values

# Create the interpolation function based on unique x-values and mean y-values
baseline_interp = interp1d(X_unique, y_mean, kind='linear', fill_value='extrapolate')

# Baseline prediction on test data without anomalies
baseline_prediction_no_anomaly = baseline_interp(X_test.flatten())

# Baseline prediction on test data with anomalies
baseline_prediction_anomalous = baseline_interp(X_test.flatten())

# 7. GPR Model Prediction for Both Test Datasets
# Predictions are the same for both datasets
y_pred, y_var = model.predict(X_test)
y_std = np.sqrt(y_var)

# 8. Model Evaluation Metrics for Both Test Datasets

# Function to compute metrics
def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

# Baseline model metrics
r2_baseline_no_anomaly, rmse_baseline_no_anomaly, mae_baseline_no_anomaly = compute_metrics(
    y_test_no_anomaly.flatten(), baseline_prediction_no_anomaly)
r2_baseline_anomalous, rmse_baseline_anomalous, mae_baseline_anomalous = compute_metrics(
    y_test_anomalous.flatten(), baseline_prediction_anomalous)

# GPR model metrics
r2_gpr_no_anomaly, rmse_gpr_no_anomaly, mae_gpr_no_anomaly = compute_metrics(
    y_test_no_anomaly.flatten(), y_pred.flatten())
r2_gpr_anomalous, rmse_gpr_anomalous, mae_gpr_anomalous = compute_metrics(
    y_test_anomalous.flatten(), y_pred.flatten())

# 9. Statistical Validation using Paired t-Test for Both Test Datasets

# Significance level
alpha = 0.05

# Function to perform t-test
def perform_ttest(y_true, baseline_pred, gpr_pred):
    residual_baseline = y_true.flatten() - baseline_pred.flatten()
    residual_gpr = y_true.flatten() - gpr_pred.flatten()
    t_stat, p_value = mannwhitneyu(residual_baseline, residual_gpr)
    return t_stat, p_value

# t-test for test dataset without anomalies
t_stat_no_anomaly, p_value_no_anomaly = perform_ttest(
    y_test_no_anomaly, baseline_prediction_no_anomaly, y_pred)

# t-test for test dataset with anomalies
t_stat_anomalous, p_value_anomalous = perform_ttest(
    y_test_anomalous, baseline_prediction_anomalous, y_pred)

# 10. Anomaly Detection only on Test Dataset with Anomalies
k = 1.5  # Threshold factor
n = 5  # Number of consecutive points

# Compute deviations for test dataset with anomalies
deviations_anomalous = np.abs(y_test_anomalous.flatten() - y_pred.flatten())

# Adjusted criterion for anomaly detection
anomalies_adjusted = np.zeros_like(deviations_anomalous, dtype=bool)
for i in range(len(X_test) - n + 1):
    window_deviation = deviations_anomalous[i:i + n]
    window_std = y_std.flatten()[i:i + n]
    individual_exceeds = window_deviation > k * window_std
    mean_condition = np.mean(window_deviation) > k * np.mean(window_std)
    if mean_condition and np.any(individual_exceeds):
        anomalies_adjusted[i:i + n] = True

# 11. Visualization in Subplots
fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)

# Subplot 1: Test dataset without anomalies
ax = axes[0]
ax.scatter(X_train_combined, y_train_combined, color='blue', label='Training Data', s=10)
ax.plot(X_test, y_test_no_anomaly, 'r-', label='Test Data without Anomalies')
ax.plot(X_test, baseline_prediction_no_anomaly, 'g--', label='Baseline Prediction')
ax.plot(X_test, y_pred, 'k-', label='GPR Prediction')
ax.fill_between(
    X_test.ravel(),
    y_pred.flatten() - k * y_std.flatten(),
    y_pred.flatten() + k * y_std.flatten(),
    alpha=0.2,
    color='gray',
    label=f'{k}-Sigma Confidence Interval'
)
#ax_twin = ax.twinx()
#ax_twin.plot(X_test, y_std, 'm--', label='Standard Deviation')
#ax_twin.set_ylabel('Standard Deviation')
#ax_twin.legend(loc='upper right')
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend(loc='upper left')
ax.set_title("Test Dataset without Anomalies")

# Subplot 2: Test dataset with anomalies
ax = axes[1]
ax.scatter(X_train_combined, y_train_combined, color='blue', label='Training Data', s=10)
ax.plot(X_test, y_test_anomalous, 'r-', label='Test Data with Anomalies')
ax.plot(X_test, baseline_prediction_anomalous, 'g--', label='Baseline Prediction')
ax.plot(X_test, y_pred, 'k-', label='GPR Prediction')
ax.fill_between(
    X_test.ravel(),
    y_pred.flatten() - k * y_std.flatten(),
    y_pred.flatten() + k * y_std.flatten(),
    alpha=0.2,
    color='gray',
    label=f'{k}-Sigma Confidence Interval'
)
#ax_twin = ax.twinx()
#ax_twin.plot(X_test, y_std, 'm--', label='Standard Deviation')
#ax_twin.set_ylabel('Standard Deviation')
#ax_twin.legend(loc='upper right')
# Highlight detected anomalies
ax.scatter(X_test[anomalies_adjusted], y_test_anomalous[anomalies_adjusted],
           color='red', label='Detected Anomalies', s=50, marker='x')
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend(loc='upper left')
ax.set_title("Test Dataset with Anomalies")

plt.tight_layout()
plt.show()

# 12. Output the Model Evaluation Metrics and t-Test Results

print("\nModel Evaluation Metrics for Test Dataset without Anomalies:")
print(f"Baseline Model - R²: {r2_baseline_no_anomaly:.4f}, RMSE: {rmse_baseline_no_anomaly:.4f}, MAE: {mae_baseline_no_anomaly:.4f}")
print(f"GPR Model     - R²: {r2_gpr_no_anomaly:.4f}, RMSE: {rmse_gpr_no_anomaly:.4f}, MAE: {mae_gpr_no_anomaly:.4f}")

print("\nPaired t-Test Results for Test Dataset without Anomalies:")
print(f"t-statistic: {t_stat_no_anomaly:.4f}, p-value: {p_value_no_anomaly:.4e}")
if p_value_no_anomaly < alpha:
    print("Result: The difference is statistically significant (reject null hypothesis).")
else:
    print("Result: The difference is not statistically significant (fail to reject null hypothesis).")

print("\nModel Evaluation Metrics for Test Dataset with Anomalies:")
print(f"Baseline Model - R²: {r2_baseline_anomalous:.4f}, RMSE: {rmse_baseline_anomalous:.4f}, MAE: {mae_baseline_anomalous:.4f}")
print(f"GPR Model     - R²: {r2_gpr_anomalous:.4f}, RMSE: {rmse_gpr_anomalous:.4f}, MAE: {mae_gpr_anomalous:.4f}")

print("\nPaired t-Test Results for Test Dataset with Anomalies:")
print(f"t-statistic: {t_stat_anomalous:.4f}, p-value: {p_value_anomalous:.4e}")
if p_value_anomalous < alpha:
    print("Result: The difference is statistically significant (reject null hypothesis).")
else:
    print("Result: The difference is not statistically significant (fail to reject null hypothesis).")
