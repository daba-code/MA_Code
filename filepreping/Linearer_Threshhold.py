import pandas as pd
import numpy as np
import glob
import os

def load_data(file_directory):
    file_paths = glob.glob(f"{file_directory}/*.csv")
    all_measurements = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=";", header=None)
        all_measurements.append((df, file_path))  # Speichern von DataFrame und Dateipfad als Tupel
    return all_measurements

def save_filtered_data(data, original_path, output_directory):
    # Generiere den neuen Dateipfad im Ausgabeordner
    base_name = os.path.basename(original_path)
    output_path = os.path.join(output_directory, base_name)
    data.to_csv(output_path, sep=";", index=False, header=False)
    print(f"Gespeichert: {output_path}")

# Funktion zum Filtern der Höhenwerte außerhalb des zulässigen Bereichs
def filter_values_outside_range(df, lower_bound=250, upper_bound=520):
    # Setze alle y-Werte außerhalb des Bereichs [250, 520] auf NaN
    df_filtered = df.applymap(lambda x: np.nan if x < lower_bound or x > upper_bound else x)
    return df_filtered

def process_all_files(file_directory, output_directory):
    all_measurements = load_data(file_directory)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Erstellen des Ausgabeordners, falls er nicht existiert
    
    processed_data = []
    
    for df, file_path in all_measurements:
        # Filtern der Werte außerhalb des Bereichs 250-520
        df_filtered = filter_values_outside_range(df)

        # Speichere die bearbeiteten Daten
        save_filtered_data(df_filtered, file_path, output_directory)
        
        # Optional: Hinzufügen zur Liste, falls weitere Verarbeitung gewünscht ist
        processed_data.append(df_filtered)

    return processed_data

# Beispiel für die Anwendung
file_directory = r"B:\dataset_slicing\optimized_files"
output_directory = r"B:\filtered_output_NaN_TH"  # Zielverzeichnis für die gefilterten Dateien
processed_data = process_all_files(file_directory, output_directory)

# Anzeigen des bereinigten DataFrames der ersten Datei
print("Bereinigte Daten (erste Datei):")
print(processed_data[0].head())
