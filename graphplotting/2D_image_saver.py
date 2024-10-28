import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob

# Laden und Umwandeln der Daten in Heatmap-Bilder
file_directory = r"B:\filtered_output_NaN_TH\sortiert\nok"
file_paths = glob.glob(f"{file_directory}/*.csv")

for idx, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path, sep=";", header=None)
    data = df.values

    # Erstellen der Graustufen-Heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='gray_r', aspect='auto')
    plt.axis('off')  # Entferne Achsen

    # Speichern als Bild
    output_image_path = f"output_image_{idx}.png"
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Zum Laden des Bildes und Weiterverarbeitung
    img = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)
    # Füge hier Bildverarbeitungsmethoden oder Klassifikationen hinzu
