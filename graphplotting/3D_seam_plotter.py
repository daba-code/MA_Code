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
        Creates widgets for the GUI including the plot canvas.
        """
        # Canvas for plotting
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Plot the entire weld seam
        self.plot_heatmap()

    def plot_heatmap(self):
        """
        Plots the entire 3D geometry of the weld seam as a 3D surface plot,
        focusing on the height range between 150 and 250 using PowerNorm with gamma.
        """
        if not self.dataframes:
            return

        # Assuming we are loading and visualizing only the first file for simplicity
        df = self.dataframes[0]
        file_path = self.file_paths[0]

        if df.empty:
            self.ax.clear()
            self.ax.set_title("No Data Available", fontsize=14)
            self.canvas.draw()
            return

        # Convert DataFrame to 2D numpy array for plotting
        seam_data = df.values

        # Prepare grid for 3D plotting
        x = np.arange(seam_data.shape[1])  # Positions along the weld seam (columns)
        y = np.arange(seam_data.shape[0])  # Profile slices (rows)
        X, Y = np.meshgrid(x, y)
        Z = seam_data  # Height values

        # Clear the current axes and set up 3D plot
        self.ax.clear()
        self.ax = self.figure.add_subplot(111, projection='3d')

        # Set the range of height values to focus between 150 and 250
        vmin, vmax = 110, 290

        # Apply PowerNorm with gamma to adjust color scaling
        gamma = 1.2 # Emphasize lower values
        norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

        # Plot the surface with the non-linear color mapping (PowerNorm)
        surface = self.ax.plot_surface(X, Y, Z, cmap='plasma_r', edgecolor='none', norm=norm)

        # Set labels and title with increased label padding
        self.ax.set_title(f'3D Weld Seam Geometry', fontsize=14)
        self.ax.set_xlabel('Pixel Number', fontsize=12, labelpad=20)  # Increase padding here
        self.ax.set_ylabel('Profile Number (cross-section)', fontsize=12, labelpad=20)  # Increase padding here
        self.ax.set_zlabel('Height', fontsize=12, labelpad=5)  # Increase padding here

        # Adjust the view angle for better visualization
        self.ax.view_init(elev=90, azim=0)

        # Remove the 2D grid (outer bounding box and grid lines)
        self.ax.xaxis.pane.set_visible(False)  # Hide the X-axis pane (background)
        self.ax.yaxis.pane.set_visible(False)  # Hide the Y-axis pane (background)
        self.ax.zaxis.pane.set_visible(False)  # Hide the Z-axis pane (background)

        # Remove gridlines along the axes
        self.ax.grid(False)

        # Add color bar for height values
        cbar = self.figure.colorbar(surface, ax=self.ax, shrink=0.5, aspect=5)
        cbar.set_label(f'Height Values(Gamma={gamma})', fontsize=12)

        self.canvas.draw()


def main():
    root = tk.Tk()
    app = ProfilePlotter(root)
    root.mainloop()

if __name__ == "__main__":
    main()
