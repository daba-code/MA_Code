import numpy as np
import matplotlib.pyplot as plt
import GPy

# 1. Erstellen der Trainingsdaten
X_train_simple = np.array([[1], [3], [5], [6], [7]])

# Erste Menge von y-Werten (Sinus-Funktion)
y_train_simple_1 = np.sin(X_train_simple)

# Zweite Menge von y-Werten mit progressiver Verschiebung
shift = np.linspace(0.1, 1.0, len(X_train_simple))  # Progressive Verschiebung
y_train_simple_2 = np.sin(X_train_simple) + shift.reshape(-1, 1)

# Kombinieren der beiden Trainingssätze
X_train_combined = np.vstack([X_train_simple, X_train_simple])
y_train_combined = np.vstack([y_train_simple_1, y_train_simple_2])

# 2. Testdaten für die Vorhersage
X_test_simple = np.linspace(0, 10, 100).reshape(-1, 1)

# 3. Definieren des Kernels (RBF-Kernel für glatte Funktionen)
kernel_simple = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)

# 4. Erstellen und Trainieren des GPRegression-Modells
model_simple = GPy.models.GPRegression(X_train_combined, y_train_combined, kernel_simple)

# Optimierung des Modells
model_simple.optimize()

# 5. Vorhersagen für die Testdaten
y_pred_simple, y_var_simple = model_simple.predict(X_test_simple)
y_std_simple = np.sqrt(y_var_simple)  # Standardabweichung für das Konfidenzintervall

# 6. Visualisierung des GPR-Modells
plt.figure(figsize=(10, 6))

# Wahre Funktion
plt.plot(X_test_simple, np.sin(X_test_simple), 'r--', label="Wahre Funktion (sin(x))")

# GPR Vorhersage
plt.plot(X_test_simple, y_pred_simple, 'b-', label="GPR Vorhersage")

# Konfidenzintervall (95%)
plt.fill_between(X_test_simple.ravel(),
                 y_pred_simple.ravel() - 1.96 * y_std_simple.ravel(),
                 y_pred_simple.ravel() + 1.96 * y_std_simple.ravel(),
                 alpha=0.2, color='blue', label="95% Konfidenzintervall")

# Plot der Trainingsdaten
plt.scatter(X_train_simple, y_train_simple_1, color='black', label="Trainingsdaten Satz 1")
plt.scatter(X_train_simple, y_train_simple_2, color='orange', label="Trainingsdaten Satz 2 (mit Verschiebung)")

plt.title("GPR Modell mit progressiver Verschiebung im zweiten Trainingssatz")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
