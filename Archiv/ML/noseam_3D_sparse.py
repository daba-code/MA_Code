import numpy as np
import GPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed

# Parameter für die Datenstruktur
num_profiles = 300     # Anzahl der Profile entlang der x-Achse
heights_per_profile = 150  # Höhenwerte pro Profil entlang der y-Achse

# Erzeugen des Gitters für die Profile (x) und Höhen (y)
x = np.arange(num_profiles)
y = np.arange(heights_per_profile)
X, Y = np.meshgrid(x, y, indexing='ij')

# Funktion zur Generierung eines glatteren Höhenprofils
def generate_height_profile(x, y, noise_level=5, amplitude=500, shift=0):
    base_wave = amplitude * (np.sin(x * 0.05) + 0.5 * np.cos(y * 0.1)) + shift
    noise = np.random.normal(0, noise_level, size=x.shape)
    heights = base_wave + noise
    return np.clip(heights, 0, amplitude)

# Erzeugen der drei Datensätze mit glattem Höhenprofil und minimalem Rauschen
Z1 = generate_height_profile(X, Y, noise_level=1, shift=0)
Z2 = generate_height_profile(X, Y, noise_level=1, shift=50)
Z3 = generate_height_profile(X, Y, noise_level=1, shift=-50)

# Zusammenführen der Daten aus den drei "Dateien" für das Training
X_flat = np.tile(X.flatten(), 3).reshape(-1, 1)
Y_flat = np.tile(Y.flatten(), 3).reshape(-1, 1)
Z_flat = np.concatenate([Z1.flatten(), Z2.flatten(), Z3.flatten()]).reshape(-1, 1)

# Zusammengefügte Eingabedaten und Ziel-Daten
input_data = np.hstack((X_flat, Y_flat))
target_data = Z_flat

# Inducing Points definieren
num_inducing = 500  # Niedrigere Anzahl für schnellere Berechnungen
kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=5.)
model = GPy.models.SparseGPRegression(input_data, target_data, kernel, num_inducing=num_inducing)

# Schritt 3: Vorhersage mit paralleler Verarbeitung
X_pred, Y_pred = np.meshgrid(x, y, indexing='ij')
X_pred_flat = X_pred.flatten().reshape(-1, 1)
Y_pred_flat = Y_pred.flatten().reshape(-1, 1)
new_input = np.hstack((X_pred_flat, Y_pred_flat))

# Funktion für parallele Vorhersageberechnung
def predict_in_batches(input_batch):
    return model.predict(input_batch)

# Aufteilen des Eingabegitters in Batches für parallele Verarbeitung
batch_size = 2000
input_batches = [new_input[i:i+batch_size] for i in range(0, new_input.shape[0], batch_size)]

# Parallele Berechnung der Vorhersagen mit joblib
results = Parallel(n_jobs=-1)(delayed(predict_in_batches)(batch) for batch in input_batches)

# Ergebnisse zusammenführen
Z_pred = np.concatenate([res[0] for res in results]).reshape(X_pred.shape)
Z_pred_var = np.concatenate([res[1] for res in results]).reshape(X_pred.shape)
Z_pred_std = np.sqrt(Z_pred_var)

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
