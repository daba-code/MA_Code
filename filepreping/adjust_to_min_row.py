import os
import pandas as pd

def trim_csv_files(directory):
    # Liste der .csv-Dateien im Verzeichnis
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    
    if not csv_files:
        raise ValueError("Keine CSV-Dateien im Verzeichnis gefunden")
    
    # Initialisierung der minimalen Zeilenanzahl mit einem großen Wert
    min_rows = float('inf')
    
    # Finde die minimale Zeilenanzahl unter allen Dateien
    for file in csv_files:
        df = pd.read_csv(os.path.join(directory, file), delimiter=';', skip_blank_lines=True)
        df = df.dropna()  # Entferne eventuell verbleibende leere Zeilen
        min_rows = min(min_rows, len(df))
    
    # Kürze alle Dateien auf die minimale Zeilenanzahl und speichere sie
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(directory, file), delimiter=';', skip_blank_lines=True)
        df = df.dropna()  # Entferne eventuell verbleibende leere Zeilen
        if len(df) > min_rows:
            df = df.iloc[:min_rows]  # Kürze auf die minimale Zeilenanzahl
        dataframes.append((file, df))
        df.to_csv(os.path.join(directory, file), index=False, sep=';', header=False)  # Speichert gekürzte DataFrames
    
    print(f"Alle CSV-Dateien wurden auf {min_rows} Zeilen gekürzt und gespeichert.\n")

    # Überprüfung der Zeilenanzahl jeder Datei nach dem Kürzen
    print("Überprüfung der Zeilenanzahl nach dem Kürzen:")
    for file, df in dataframes:
        print(f"Datei {file} hat {len(df)} Zeilen.")

# Beispielverwendung
verzeichnis = r'B:\test_files_1dim_v3'
trim_csv_files(verzeichnis)
