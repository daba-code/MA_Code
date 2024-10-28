import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from py_helpers import clear_empty_profiles  # Assuming this function is defined elsewhere
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from matplotlib.colors import PowerNorm

class ProfilePlotter:
    def __init__(self, root):
        self.root = root
        self.root.title("Weld Seam Visualization")

        # File selection
        self.file_paths = self.select_files()
        self.dataframes = [self.load_and_clean_file(file) for file in self.file_paths]
        self.current_profile_index = 0  # Start with the first profile

        # GUI setup
        self.create_widgets()

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
            df_clean = clear_empty_profiles(df)  # Remove empty profiles (assumed to be rows with NaN or empty values)
            print(f"Loaded {file_path} with {df_clean.shape[0]} profiles.")
            return df_clean
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return pd.DataFrame()

    def create_widgets(self):
        """
        Creates widgets for the GUI including the plot canvas and navigation buttons.
        """
        # Canvas for plotting
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Button frame for navigation
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Navigation buttons
        self.prev_button = tk.Button(self.button_frame, text="Previous Profile", command=self.previous_profile)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.next_button = tk.Button(self.button_frame, text="Next Profile", command=self.next_profile)
        self.next_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Initial plot of the first profile
        self.plot_heatmap(self.current_profile_index)

    def plot_heatmap(self, profile_index):
        """
        Plots the 3D geometry of the weld seam for a specific profile.
        """
        if not self.dataframes:
            return

        df = self.dataframes[profile_index]
        file_path = self.file_paths[profile_index]

        if df.empty:
            self.ax.clear()
            self.ax.set_title("No Data Available", fontsize=14)
            self.canvas.draw()
            return

        seam_data = df.values
        x = np.arange(seam_data.shape[1])
        y = np.arange(seam_data.shape[0])
        X, Y = np.meshgrid(x, y)
        Z = seam_data

        # Clear current axes and set up a new 3D plot
        self.ax.clear()
        self.ax = self.figure.add_subplot(111, projection='3d')

        # Dynamically adjust color normalization based on the min and max values in the profile
        vmin, vmax = Z.min(), Z.max()
        gamma = 1  # Adjust gamma as needed
        norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

        # Plot the surface with adjusted color mapping
        surface = self.ax.plot_surface(X, Y, Z, cmap='nipy_spectral', edgecolor='none', norm=norm)

        # Set labels and title with increased padding
        self.ax.set_title(f'3D Weld Seam Geometry - File {profile_index + 1}', fontsize=14)
        self.ax.set_xlabel('Pixel Number', fontsize=12, labelpad=20)
        self.ax.set_ylabel('Profile Number (cross-section)', fontsize=12, labelpad=20)
        self.ax.set_zlabel('Height', fontsize=12, labelpad=5)

        # Adjust the view angle
        self.ax.view_init(elev=25, azim=270)

        # Remove the background grid and panes
        self.ax.xaxis.pane.set_visible(False)
        self.ax.yaxis.pane.set_visible(False)
        self.ax.zaxis.pane.set_visible(False)
        self.ax.grid(False)

        # Add color bar for height values
        cbar = self.figure.colorbar(surface, ax=self.ax, shrink=0.5, aspect=5)
        cbar.set_label(f'Height Values (Gamma={gamma})', fontsize=12)

        self.canvas.draw()

    def next_profile(self):
        """Moves to the next profile and updates the plot."""
        self.current_profile_index = (self.current_profile_index + 1) % len(self.dataframes)
        self.plot_heatmap(self.current_profile_index)

    def previous_profile(self):
        """Moves to the previous profile and updates the plot."""
        self.current_profile_index = (self.current_profile_index - 1) % len(self.dataframes)
        self.plot_heatmap(self.current_profile_index)

def main():
    root = tk.Tk()
    app = ProfilePlotter(root)
    root.mainloop()

if __name__ == "__main__":
    main()
