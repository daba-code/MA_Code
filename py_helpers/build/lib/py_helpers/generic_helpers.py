#generic_helpers.py
import tkinter as tk
from tkinter import filedialog

def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory()  # Open the dialog to choose a directory
    root.destroy()  # Ensure the root tkinter instance is destroyed after selection
    return folder_selected