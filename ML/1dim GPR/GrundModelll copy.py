import numpy as np
import matplotlib.pyplot as plt
import GPy
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.stats as stats
from scipy.stats import ttest_rel, mannwhitneyu, shapiro
import matplotlib.cm as cm

np.random.seed(42)  # Setzt den Zufallsseed für Reproduzierbarkeit

# 1. Erstellen der Trainingsdaten mit variabler Streuung
X_train = np.linspace(0, 10, 100).reshape(-1, 1)  # 100 x-Werte von 0 bis 10

# Definieren der variablen Streuung entlang der x-Achse (z.B. zunehmende Streuung)
noise_levels = np.linspace(1, 5, 100)  # Rauschlevel von 1 bis 5

# y-Werte mit variabler Streuung erzeugen
y_train_variations = [np.sin(X_train) + np.random.normal(0, noise_levels[:, None], X_train.shape) for _ in range(10)]

# Kombinieren der X- und y-Werte
X_train_combined = np.vstack([X_train for _ in range(10)])  # X-Werte für jeden der 10 y-Datensätze duplizieren
y_train_combined = np.vstack(y_train_variations).flatten()  # Stapeln aller y-Werte in einem Vektor

# 2. Testdaten für die Vorhersage und als Referenzkurve
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_test = np.sin(X_test).ravel()  # Sinus als Referenzfunktion

# 3. Definieren des Kernels in GPy
kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)  # RBF-Kernel für glatte Funktionen

#kernel.variance.constrain_bounded(0.1, 1)  # Variance zwischen 0.1 und 100
#kernel.lengthscale.constrain_bounded(0.1, 0.7)  # Lengthscale zwischen 0.1 und 50
#kernel.power.constrain_bounded(0.01, 2)  # Power zwischen 0.01 und 2

# 4. GPR-Modell erstellen und trainieren
model = GPy.models.GPRegression(X_train_combined, y_train_combined.reshape(-1, 1), kernel)
#model.Gaussian_noise.variance = 0.1  # Setzen der Rauschvarianz

# Optimierung des Modells
model.optimize(messages=True, max_iters=1000)


print("Optimierte Kernel-Parameter:")
print(f"Variance: {model.kern.variance.values[0]}")
print(f"Lengthscale: {model.kern.lengthscale.values[0]}")
#print(f"Power: {model.kern.power.values[0]}")

print("Optimierte Rauschvarianz:")
print(f"Noise variance: {model.Gaussian_noise.variance.values[0]}")

# Vorhersagen für die Testdaten
y_pred, y_var = model.predict(X_test)
y_std = np.sqrt(y_var)  # Standardabweichung für Konfidenzintervall

# 5. Baseline Modell: Durchschnitt über die Trainingsdatensätze für jeden x-Wert
X_unique = np.unique(X_train_combined)  # Einzigartige x-Werte
# Für jeden einzigartigen x-Wert den Mittelwert berechnen (ohne Interpolation)
y_mean_per_x = np.array([np.mean(y_train_combined[X_train_combined.flatten() == x]) for x in X_unique])

# Berechnung der Vorhersagen für das Baseline-Modell
y_pred_baseline = np.array([y_mean_per_x[np.argmin(np.abs(X_unique - x))] for x in X_test.flatten()])

# 7. Berechnung der Metriken
def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

# Berechnung der Modellmetriken
r2_gpr, rmse_gpr, mae_gpr = compute_metrics(y_test, y_pred)

# Ausgabe der Metriken
print(f"GPR-Modell Metriken:")
print(f"R²: {r2_gpr:.4f}")
print(f"RMSE: {rmse_gpr:.4f}")
print(f"MAE: {mae_gpr:.4f}")

# Berechnung der Metriken für das Baseline-Modell
r2_baseline, rmse_baseline, mae_baseline = compute_metrics(y_test, y_pred_baseline)

# Ausgabe der Baseline-Metriken
print(f"\nBaseline-Modell Metriken:")
print(f"R²: {r2_baseline:.4f}")
print(f"RMSE: {rmse_baseline:.4f}")
print(f"MAE: {mae_baseline:.4f}")

# 8. Statistischer Test (Mann-Whitney-U-Test für Residuen)
residuals_gpr = y_test - y_pred.flatten()
residuals_baseline = y_test - y_pred_baseline.flatten()

t_stat, p_value = mannwhitneyu(residuals_baseline, residuals_gpr)

# Ausgabe des Mann-Whitney-U-Test Ergebnisses
print(f"\nMann-Whitney-U-Test Ergebnis:")
print(f"T-Statistik: {t_stat:.4f}")
print(f"P-Wert: {p_value:.4f}")

if p_value < 0.1:
    print("Der Unterschied zwischen GPR und Baseline-Modell ist statistisch signifikant (p < 0.1).")
else:
    print("Der Unterschied zwischen GPR und Baseline-Modell ist nicht statistisch signifikant (p >= 0.1).")

# 9. Shapiro-Wilk-Test zur Überprüfung der Normalverteilung der Residuen
stat_gpr, p_gpr = shapiro(residuals_gpr)
stat_baseline, p_baseline = shapiro(residuals_baseline)

print("\nShapiro-Wilk-Test Ergebnisse:")
print(f"GPR-Residuen: W-Statistik: {stat_gpr:.4f}, P-Wert: {p_gpr:.4f}")
print(f"Baseline-Residuen: W-Statistik: {stat_baseline:.4f}, P-Wert: {p_baseline:.4f}")

# Erstellen eines Subplots mit 1 Zeile und 2 Spalten
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 8.27))  # Größe auf A4 angepasst (landscape)

# Q-Q-Plot für GPR-Residuen (linker Plot)
stats.probplot(residuals_gpr, dist="norm", plot=ax1)
ax1.set_title("Q-Q-Plot residuals GPR model", fontsize=14)
ax1.set_xlabel("Theoretical Quantiles", fontsize=12)
ax1.set_ylabel("Empirical Quantiles", fontsize=12)
ax1.grid(True)

# Q-Q-Plot für Baseline-Residuen (rechter Plot)
stats.probplot(residuals_baseline, dist="norm", plot=ax2)
ax2.set_title("Q-Q-Plot residuals Baseline model", fontsize=14)
ax2.set_xlabel("Theoretical Quantiles", fontsize=12)
ax2.set_ylabel("Empirical Quantiles", fontsize=12)
ax2.grid(True)

# Platz für beide Plots optimieren
plt.tight_layout()

# Plot anzeigen
plt.show()

# 6. Plotten der Ergebnisse
fig, ax1 = plt.subplots(figsize=(12, 6))

# GPR Vorhersage plotten
ax1.plot(X_test, y_pred, 'k-', label='GPy predict', linewidth=2)

# Konfidenzintervall für die GPR Vorhersage
#ax1.fill_between(
#    X_test.ravel(),
#    y_pred.flatten() - 1.96 * y_std.flatten(),
#    y_pred.flatten() + 1.96 * y_std.flatten(),
#    alpha=0.2,
#    color='gray',
#    label='95% Konfidenzintervall'
#)

# Plot der Referenzsinuskurve (Testdaten)
ax1.plot(X_test, y_test, 'r-', label='testfile (sin)', linewidth=2)

# Plot der Baseline Vorhersage
ax1.plot(X_test, y_pred_baseline, 'g--', label='Baseline predict', linewidth=2)

# Trainingsdatenpunkte als Streuwolke plotten
for idx, y_train in enumerate(y_train_variations):
    ax1.scatter(X_train, y_train, alpha=0.3, label=f'training points' if idx == 0 else "", s=10)

# Achsen beschriften und Formatierung
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)", color='black')
ax1.legend(loc='upper left')

# Plot anzeigen
plt.show()
