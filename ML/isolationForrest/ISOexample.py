import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Beispiel: U-förmiges Profil für Trainingsdaten
x_train = np.linspace(0, 10, 100).reshape(-1, 1)  # x-Werte: Positionen entlang der Schweißnaht
y_train = 70 - (x_train - 5)**2 + np.random.normal(0, 2, x_train.shape)  # U-förmiges Profil mit Rauschen

# Beispiel: Testdaten mit einer Anomalie
x_test = np.linspace(0, 10, 100).reshape(-1, 1)  # Gleiche x-Werte wie im Training
y_test = 70 - (x_test - 5)**2 + np.random.normal(0, 2, x_test.shape)
y_test[80] = 90  # Künstliche Anomalie (z.B. an einem Rand)

# Isolation Forest zur Anomalieerkennung
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(y_train)  # Training auf den y-Werten des U-Profils

# Vorhersage für Testdaten: -1 bedeutet Anomalie, 1 bedeutet normal
y_pred = clf.predict(y_test)

# Visualisierung der Ergebnisse
plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, label='Trainingsdaten (U-förmiges Profil)', color='blue')
plt.plot(x_test, y_test, label='Testdaten (mit Anomalie)', color='orange')
plt.scatter(x_test[y_pred == -1], y_test[y_pred == -1], color='red', label='Anomalie', s=100)
plt.axhline(np.mean(y_train), color='gray', linestyle='--', label='Durchschnitt')
plt.title('Anomalieerkennung bei U-förmigem Schweißnahtprofil')
plt.xlabel('Position entlang der Schweißnaht (x-Wert)')
plt.ylabel('Höhe der Schweißnaht (y-Wert)')
plt.legend()
plt.show()

print("Vorhersagen:", y_pred)  # Ausgabe: 1 = normal, -1 = Anomalie
