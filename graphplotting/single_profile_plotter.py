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
        
        # Bind arrow keys to the root window
        self.root.bind("<Left>", self.previous_profile_event)
        self.root.bind("<Right>", self.next_profile_event)

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
        Creates widgets for the GUI including navigation buttons and a plot canvas.
        """
        # Canvas for plotting
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Navigation buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.prev_button = tk.Button(self.button_frame, text="Previous Profile", command=self.previous_profile)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.next_button = tk.Button(self.button_frame, text="Next Profile", command=self.next_profile)
        self.next_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Bind mouse button hold for fast scrolling
        self.prev_button.bind("<ButtonPress-1>", lambda event: self.start_fast_scroll(-1))
        self.next_button.bind("<ButtonPress-1>", lambda event: self.start_fast_scroll(1))
        self.prev_button.bind("<ButtonRelease-1>", self.stop_fast_scroll)
        self.next_button.bind("<ButtonRelease-1>", self.stop_fast_scroll)

        # Initial plot
        self.plot_profile()

    def plot_profile(self):
        """
        Plots the current profile from the current file.
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
        # Plot
        self.ax.clear()
        self.ax.plot(x_values, data_values, color = color, linewidth = 4, label=f'Profile {self.current_profile_index + 1}')
        self.ax.set_ylim(0, 800)
        self.ax.set_title(f'Profile {self.current_profile_index + 1} from {file_path}', fontsize=0)
        self.ax.set_xlabel('column', fontsize=40)
        self.ax.set_ylabel('height values', fontsize=40)
        
         # Customize tick label font sizes
        self.ax.tick_params(axis='x', labelsize=30)  # X-axis tick font size
        self.ax.tick_params(axis='y', labelsize=30)  # Y-axis tick font size

        self.ax.legend()
        self.canvas.draw()

    def next_profile(self):
        """
        Moves to the next profile and updates the plot.
        """
        df = self.dataframes[self.current_file_index]
        if df.empty:
            return
        
        self.current_profile_index += 1
        if self.current_profile_index >= df.shape[0]:
            self.current_profile_index = 0
            self.current_file_index = (self.current_file_index + 1) % len(self.dataframes)

        self.plot_profile()

    def previous_profile(self):
        """
        Moves to the previous profile and updates the plot.
        """
        df = self.dataframes[self.current_file_index]
        if df.empty:
            return
        
        self.current_profile_index -= 1
        if self.current_profile_index < 0:
            self.current_file_index = (self.current_file_index - 1) % len(self.dataframes)
            df = self.dataframes[self.current_file_index]
            self.current_profile_index = df.shape[0] - 1

        self.plot_profile()

    def next_profile_event(self, event):
        """
        Event handler for moving to the next profile using the right arrow key.
        """
        self.start_fast_scroll(1)
        self.root.after(self.scroll_delay, self.stop_fast_scroll)

    def previous_profile_event(self, event):
        """
        Event handler for moving to the previous profile using the left arrow key.
        """
        self.start_fast_scroll(-1)
        self.root.after(self.scroll_delay, self.stop_fast_scroll)

    def start_fast_scroll(self, direction):
        """
        Starts the fast scrolling process in a given direction.

        Parameters:
        - direction: int, direction of scrolling (+1 for next, -1 for previous).
        """
        if not self.fast_scroll:
            self.fast_scroll = True
            self.fast_scroll_loop(direction)

    def stop_fast_scroll(self, event=None):
        """
        Stops the fast scrolling process.
        """
        self.fast_scroll = False

    def fast_scroll_loop(self, direction):
        """
        Handles the fast scrolling loop by repeatedly changing profiles.

        Parameters:
        - direction: int, direction of scrolling (+1 for next, -1 for previous).
        """
        if self.fast_scroll:
            if direction > 0:
                self.next_profile()
            else:
                self.previous_profile()
            self.root.after(self.scroll_delay, lambda: self.fast_scroll_loop(direction))

def main():
    root = tk.Tk()
    app = ProfilePlotter(root)
    root.mainloop()

if __name__ == "__main__":
    main()
