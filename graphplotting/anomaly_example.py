import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

# Generate synthetic data for "normal" instances
normal_data = np.random.randn(50, 2)

# Add a few "anomalies" far from the normal data
anomalies = np.array([[3, 3], [-3, -3]])

# Combine normal data and anomalies
X = np.vstack([normal_data, anomalies])

# Fit a one-class SVM for anomaly detection
oc_svm = OneClassSVM(kernel="rbf", gamma=0.05, nu=0.01)
oc_svm.fit(normal_data)

# Create a grid for the decision boundary
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
grid_points = np.c_[xx.ravel(), yy.ravel()]
decision_values = oc_svm.decision_function(grid_points).reshape(xx.shape)

# Plot the decision boundary and the data
plt.figure(figsize=(8, 6))

# Plot the decision boundary
plt.contour(xx, yy, decision_values, levels=[0], linewidths=2, colors="black", linestyles="--")

# Plot normal instances
plt.scatter(normal_data[:, 0], normal_data[:, 1], label="Normal Instances", color="black", marker="x")

# Plot anomalies
plt.scatter(anomalies[:, 0], anomalies[:, 1], label="Anomalies", color="red", s=80)

# Add labels and legend
plt.title("One-class Anomaly Detection")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
