import numpy as np
import GPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score

# Parameter für die Datenstruktur
num_profiles = 30      # Anzahl der Profile entlang der x-Achse
heights_per_profile = 20  # Höhenwerte pro Profil entlang der y-Achse

# Erzeugen des Gitters für die Profile (x) und Höhen (y)
x = np.arange(num_profiles)
y = np.arange(heights_per_profile)
X, Y = np.meshgrid(x, y, indexing='ij')

# Rauschparameter
sin_noise1, cos_noise1 = 0.05, 0.05  # Rauschen für Datei 1
sin_noise2, cos_noise2 = 0.1, 0.1    # Rauschen für Datei 2
sin_noise3, cos_noise3 = 0.15, 0.15  # Rauschen für Datei 3

# Erzeugen der drei Datensätze mit Rauschen
Z1 = (np.sin(X * 0.2) + np.cos(Y * 0.3) +
      np.random.normal(0, sin_noise1, X.shape) +
      np.random.normal(0, cos_noise1, Y.shape))

Z2 = (np.sin(X * 0.2) + np.cos(Y * 0.3) + 0.1 +
      np.random.normal(0, sin_noise2, X.shape) +
      np.random.normal(0, cos_noise2, Y.shape))

Z3 = (np.sin(X * 0.2) + np.cos(Y * 0.3) - 0.1 +
      np.random.normal(0, sin_noise3, X.shape) +
      np.random.normal(0, cos_noise3, Y.shape))

# Zusammenführen der Daten aus den drei "Dateien" für das Training
X_flat = np.tile(X.flatten(), 3).reshape(-1, 1)  # Profile (x), 3x repliziert für die Dateien
Y_flat = np.tile(Y.flatten(), 3).reshape(-1, 1)  # Höhenwerte (y), 3x repliziert für die Dateien
Z_flat = np.concatenate([Z1.flatten(), Z2.flatten(), Z3.flatten()]).reshape(-1, 1)  # Zielwerte aus allen Dateien

# Zusammengefügte Eingabedaten und Ziel-Daten
input_data = np.hstack((X_flat, Y_flat))  # Shape (1800, 2)
target_data = Z_flat                      # Shape (1800, 1)

# Training des Modells
kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=5.)
model = GPy.models.GPRegression(input_data, target_data, kernel)
model.optimize(messages=True)

# Schritt 3: Vorhersage für das gesamte Profilgitter
X_pred, Y_pred = np.meshgrid(x, y, indexing='ij')
X_pred_flat = X_pred.flatten().reshape(-1, 1)
Y_pred_flat = Y_pred.flatten().reshape(-1, 1)
new_input = np.hstack((X_pred_flat, Y_pred_flat))

# Vorhersage der Höhenwerte Z für die Profile, inklusive Varianz
Z_pred, Z_pred_var = model.predict(new_input)
Z_pred = Z_pred.reshape(X_pred.shape)  # Shape (30, 20) nach Vorhersage
Z_pred_std = np.sqrt(Z_pred_var).reshape(X_pred.shape)  # Standardabweichung der Vorhersage

# Konfidenzintervalle berechnen (95% Konfidenzintervall)
confidence_interval_lower = Z_pred - 1.96 * Z_pred_std
confidence_interval_upper = Z_pred + 1.96 * Z_pred_std

# Schritt 4: Visualisierung der 3D-Struktur, der GPR-Vorhersage und der Varianz
fig = plt.figure(figsize=(24, 10))

# Originaldaten der ersten Datei
ax1 = fig.add_subplot(151, projection='3d')
ax1.plot_surface(X, Y, Z1, cmap="viridis", edgecolor='none')
ax1.set_title("Originale Profilstruktur (Datei 1)")
ax1.set_xlabel("Profile (X)")
ax1.set_ylabel("Höhenindex (Y)")
ax1.set_zlabel("Höhenwert (Z)")

# Originaldaten der zweiten Datei
ax2 = fig.add_subplot(152, projection='3d')
ax2.plot_surface(X, Y, Z2, cmap="viridis", edgecolor='none')
ax2.set_title("Originale Profilstruktur (Datei 2)")
ax2.set_xlabel("Profile (X)")
ax2.set_ylabel("Höhenindex (Y)")
ax2.set_zlabel("Höhenwert (Z)")

# Originaldaten der dritten Datei
ax3 = fig.add_subplot(153, projection='3d')
ax3.plot_surface(X, Y, Z3, cmap="viridis", edgecolor='none')
ax3.set_title("Originale Profilstruktur (Datei 3)")
ax3.set_xlabel("Profile (X)")
ax3.set_ylabel("Höhenindex (Y)")
ax3.set_zlabel("Höhenwert (Z)")

# GPR-Vorhersage visualisieren mit Mean und Konfidenzintervall
ax4 = fig.add_subplot(154, projection='3d')
ax4.plot_surface(X_pred, Y_pred, Z_pred, cmap="viridis", edgecolor='none')
ax4.plot_surface(X_pred, Y_pred, confidence_interval_lower, color='orange', alpha=0.3, edgecolor='none')
ax4.plot_surface(X_pred, Y_pred, confidence_interval_upper, color='orange', alpha=0.3, edgecolor='none')
ax4.set_title("GPR Vorhersage mit Konfidenzintervall")
ax4.set_xlabel("Profile (X)")
ax4.set_ylabel("Höhenindex (Y)")
ax4.set_zlabel("Vorhergesagter Höhenwert (Z)")

# Varianz als separate Fläche visualisieren
ax5 = fig.add_subplot(155, projection='3d')
ax5.plot_surface(X_pred, Y_pred, Z_pred_std**2, cmap="inferno", edgecolor='none')
ax5.set_title("Varianz der Vorhersage")
ax5.set_xlabel("Profile (X)")
ax5.set_ylabel("Höhenindex (Y)")
ax5.set_zlabel("Varianz")

plt.tight_layout()
plt.show()

# RMSE und R^2 berechnen
Z1_flat = Z1.flatten()
Z_pred_flat = Z_pred.flatten()
rmse = np.sqrt(mean_squared_error(Z1_flat, Z_pred_flat))
r2 = r2_score(Z1_flat, Z_pred_flat)

print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")
