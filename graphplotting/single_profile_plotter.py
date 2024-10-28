import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from py_helpers import clear_empty_profiles  # Assuming this function is defined elsewhere

class ProfilePlotter:
    def __init__(self, root):
        self.root = root
        self.root.title("Profile Plotter")

        # File selection
        self.file_paths = self.select_files()
        self.dataframes = [self.load_and_clean_file(file) for file in self.file_paths]
        self.current_profile_index = 0
        self.current_file_index = 0

        # GUI setup
        self.create_widgets()
        
        # Faster navigation variables
        self.fast_scroll = False
        self.scroll_delay = 100  # Initial delay in milliseconds

    def select_files(self):
        """
        Opens a file dialog to select one or more CSV files for processing.
        
        Returns:
        - tuple of selected file paths.
        """
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_selected = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV Files", "*.csv")])
        root.destroy()  # Ensure the root tkinter instance is destroyed after selection
        return file_selected

    def load_and_clean_file(self, file_path):
        """
        Loads and cleans a file by removing empty profiles.

        Parameters:
        - file_path: str, path to the file.

        Returns:
        - DataFrame containing cleaned data.
        """
        try:
            df = pd.read_csv(file_path, delimiter=";")
            df_clean = clear_empty_profiles(df)
            print(f"Loaded {file_path} with {df_clean.shape[0]} profiles.")
            return df_clean
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return pd.DataFrame()

    def create_widgets(self):
        """
        Creates widgets for the GUI including navigation buttons, plot canvas, and row entry.
        """
        # Canvas for plotting
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Row selection
        self.row_entry_frame = tk.Frame(self.root)
        self.row_entry_frame.pack(side=tk.TOP, fill=tk.X)

        self.row_label = tk.Label(self.row_entry_frame, text="Enter Row Number:")
        self.row_label.pack(side=tk.LEFT, padx=5)

        self.row_entry = tk.Entry(self.row_entry_frame, width=5)
        self.row_entry.pack(side=tk.LEFT, padx=5)

        self.row_button = tk.Button(self.row_entry_frame, text="Plot Row", command=self.plot_specific_row)
        self.row_button.pack(side=tk.LEFT, padx=5)

        # Navigation buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Initial plot
        self.plot_profile()

    def plot_profile(self):
        """
        Plots the current profile from the current file with auto-scaled y-axis.
        """
        df = self.dataframes[self.current_file_index]
        file_path = self.file_paths[self.current_file_index]

        if df.empty:
            self.ax.clear()
            self.ax.set_title("No Data Available", fontsize=14)
            self.canvas.draw()
            return

        if self.current_profile_index >= df.shape[0]:
            self.current_profile_index = 0

        data_values = df.iloc[self.current_profile_index].values
        x_values = np.arange(len(data_values))
        color = "green"
        
        # Plot with automatic y-axis scaling
        self.ax.clear()
        self.ax.plot(x_values, data_values, color=color, linewidth=4, label=f'Profile {self.current_profile_index + 1}')
        self.ax.relim()  # Recalculate limits based on the data
        self.ax.autoscale(enable=True, axis='y')  # Enable auto-scaling on y-axis

        self.ax.set_title(f'Profile {self.current_profile_index + 1} from {file_path}', fontsize=20)
        self.ax.set_xlabel('Column', fontsize=20)
        self.ax.set_ylabel('Height Values', fontsize=20)

        # Customize tick label font sizes
        self.ax.tick_params(axis='x', labelsize=15)  # X-axis tick font size
        self.ax.tick_params(axis='y', labelsize=15)  # Y-axis tick font size

        self.ax.legend()
        self.canvas.draw()

    def plot_specific_row(self):
        """
        Plots the profile from the specific row number entered by the user.
        """
        try:
            row_index = int(self.row_entry.get()) - 1
            df = self.dataframes[self.current_file_index]
            if 0 <= row_index < df.shape[0]:
                self.current_profile_index = row_index
                self.plot_profile()
            else:
                print(f"Row number out of range. Please enter a number between 1 and {df.shape[0]}")
        except ValueError:
            print("Invalid row number. Please enter a valid integer.")

def main():
    root = tk.Tk()
    app = ProfilePlotter(root)
    root.mainloop()

if __name__ == "__main__":
    main()
