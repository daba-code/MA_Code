import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Verzeichnisse mit den CSV-Dateien
file_directory_ok = r"B:\filtered_output_NaN_TH\sortiert\ok"
file_directory_nok = r"B:\filtered_output_NaN_TH\sortiert\nok"
normal_files = glob.glob(os.path.join(file_directory_ok, "*.csv"))
pore_files = glob.glob(os.path.join(file_directory_nok, "*.csv"))

# Nok-Reihen, die Poren enthalten
nok_rows_with_pores = [219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 
                       231, 232, 233, 234, 235, 236, 237, 238, 277, 278, 279, 280, 
                       281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 
                       293, 294, 295, 296, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 
                       1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 
                       1092, 1093, 1094, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 
                       1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 
                       1743, 1744, 1745, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 
                       2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 
                       2365, 2366, 2367]

# Funktion zum Laden und Labeln der Dateien
def load_and_label_files(file_paths, label, nok_rows=None):
    data = []
    labels = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=";", header=None)
        for row_idx in range(df.shape[0]):
            row_data = df.iloc[row_idx].values.flatten()
            data.append(row_data)
            # Verwende `1` für Reihen mit Poren in `nok`, sonst `0`
            if nok_rows and row_idx in nok_rows:
                labels.append(1)
            else:
                labels.append(label)
    return data, labels

# Laden und Labeln der Daten
normal_data, normal_labels = load_and_label_files(normal_files, label=0)
nok_data, nok_labels = load_and_label_files(pore_files, label=0, nok_rows=nok_rows_with_pores)

# Zusammenführen der Daten
X = np.array(normal_data + nok_data)
y = np.array(normal_labels + nok_labels)

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Klassifikator erstellen
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Vorhersagen und Auswertung
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
