import os
import pandas as pd

def trim_csv_files(directory):
    # Erstelle einen Unterordner für die bearbeiteten Dateien
    output_directory = os.path.join(directory, "gekuerzt")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Liste der .csv-Dateien im Verzeichnis
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    
    if not csv_files:
        raise ValueError("Keine CSV-Dateien im Verzeichnis gefunden")
    
    # Initialisierung der minimalen Zeilenanzahl mit einem großen Wert
    min_rows = float('inf')
    
    # Finde die minimale Zeilenanzahl unter allen Dateien, nachdem NaN-Zeilen entfernt wurden
    for file in csv_files:
        df = pd.read_csv(os.path.join(directory, file), delimiter=';', skip_blank_lines=True)
        
        # Finde Zeilen, die nur aus NaN bestehen und protokolliere sie
        nan_only_rows = df[df.isna().all(axis=1)].index.tolist()
        if nan_only_rows:
            print(f"In Datei {file} wurden die folgenden vollständig leeren Zeilen gelöscht: {nan_only_rows}")
        
        # Entferne vollständig leere Zeilen (bestehend nur aus NaNs)
        df = df.dropna(how='all')
        
        # Fülle verbleibende NaNs mit den Werten aus der nächsten nicht-NaN-Zeile
        nan_rows = df[df.isna().any(axis=1)].index.tolist()
        if nan_rows:
            print(f"In Datei {file} wurden die folgenden Zeilen mit NaNs bearbeitet: {nan_rows}")
        df = df.bfill(axis=0)
        
        # Aktualisiere die minimale Zeilenanzahl
        min_rows = min(min_rows, len(df))
    
    # Kürze alle Dateien auf die minimale Zeilenanzahl, runde und speichere sie im Unterordner
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(directory, file), delimiter=';', skip_blank_lines=True)
        
        # Finde Zeilen, die nur aus NaN bestehen und protokolliere sie
        nan_only_rows = df[df.isna().all(axis=1)].index.tolist()
        if nan_only_rows:
            print(f"In Datei {file} wurden die folgenden vollständig leeren Zeilen gelöscht: {nan_only_rows}")
        
        # Entferne vollständig leere Zeilen (bestehend nur aus NaNs)
        df = df.dropna(how='all')
        
        # Fülle verbleibende NaNs mit den Werten aus der nächsten nicht-NaN-Zeile
        nan_rows = df[df.isna().any(axis=1)].index.tolist()
        if nan_rows:
            print(f"In Datei {file} wurden die folgenden Zeilen mit NaNs bearbeitet: {nan_rows}")
        df = df.bfill(axis=0)
        
        # Kürze auf die minimale Zeilenanzahl, falls erforderlich
        if len(df) > min_rows:
            df = df.iloc[:min_rows]
        
        # Runde alle Werte auf ganze Zahlen
        df = df.round(0).astype(int)
        
        # Speichert den DataFrame und den Dateinamen für die Überprüfung der Zeilenanzahl
        dataframes.append((file, df))
        
        # Speichere den gekürzten und gerundeten DataFrame im Unterordner
        output_path = os.path.join(output_directory, file)
        df.to_csv(output_path, index=False, sep=';', header=False)
        print(f"Gespeichert: {output_path}")
    
    print(f"\nAlle CSV-Dateien wurden auf {min_rows} Zeilen gekürzt, gerundet und gespeichert.\n")

    # Überprüfung der Zeilenanzahl jeder Datei nach dem Kürzen
    print("Überprüfung der Zeilenanzahl nach dem Kürzen:")
    for file, df in dataframes:
        print(f"Datei {file} hat {len(df)} Zeilen.")

# Beispielverwendung
verzeichnis = r'B:\filtered_output\gekuerzt\aligned_files\processed_files'
trim_csv_files(verzeichnis)
