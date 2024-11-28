import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Generate a manageable 3D dataset
np.random.seed(42)
x = np.tile(np.arange(1, 201), 50)  # x-values: 200 positions, repeated for 50 profiles
z = np.repeat(np.arange(1, 51), 200)  # z-values: 50 profiles
y = 5 + np.random.normal(0, 0.2, len(x))  # Normal values with slight noise

# Add anomalies: A block of elevated y-values
anomaly_indices = np.where((x >= 50) & (x <= 55) & (z >= 10) & (z <= 15))
y[anomaly_indices] += 2  # Artificial anomalies (elevated height values)

# Create a DataFrame
data = pd.DataFrame({'x': x, 'z': z, 'y': y})

# Train an Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
data['anomaly_score'] = iso_forest.fit_predict(data[['x', 'z', 'y']])
data['is_anomaly'] = data['anomaly_score'] == -1  # -1 indicates anomalies

# Visualization: Show anomalies across x and z
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatterplot for normal data
normal_data = data[~data['is_anomaly']]
ax.scatter(normal_data['x'], normal_data['z'], normal_data['y'], c='blue', label='Normal', alpha=0.7)

# Scatterplot for anomalies
anomaly_data = data[data['is_anomaly']]
ax.scatter(anomaly_data['x'], anomaly_data['z'], anomaly_data['y'], c='red', label='Anomalies', s=40, zorder=5)

# Axis labeling and title
ax.set_title('Isolation Forest: Anomalies in a 3D Data Structure', fontsize=14)
ax.set_xlabel('Position (x)', fontsize=12)
ax.set_ylabel('Profile (z)', fontsize=12)
ax.set_zlabel('Height (y)', fontsize=12)

# Legend and display
ax.legend()
plt.show()
