import cv2
import numpy as np
import matplotlib.pyplot as plt

# Bild laden
file_path = r"B:\MA_Code\output_image_0.png"  # Ersetze durch den Bildpfad
image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# Maske erstellen: Bereiche links und rechts auf 0 setzen
mask = np.ones_like(image, dtype=np.uint8)  # Starte mit einer Maske mit allen Werten auf 1
mask[:, :100] = 0   # Linke Seite ignorieren (hier die ersten 100 Spalten)
mask[:, -100:] = 0  # Rechte Seite ignorieren (hier die letzten 100 Spalten)

# Maske auf das Bild anwenden
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Canny Edge Detection mit maskiertem Bild
edges = cv2.Canny(masked_image, threshold1=20, threshold2=60)

# Originalbild, maskiertes Bild und Kanten anzeigen
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray', aspect='auto')
plt.colorbar(label='Intensity')
plt.title("Original Gray Image")

plt.subplot(1, 3, 2)
plt.imshow(masked_image, cmap='gray', aspect='auto')
plt.colorbar(label='Intensity')
plt.title("Masked Image with Ignored Sides")

plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray', aspect='auto')
plt.colorbar(label='Edge Intensity')
plt.title("Edge Detection on Masked Area")
plt.show()
