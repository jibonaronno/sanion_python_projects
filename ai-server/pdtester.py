import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import (
    Tk, Button, filedialog, Frame, Listbox,
    Scrollbar, messagebox, Label, ttk, Menu, SINGLE
)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from datetime import datetime, timedelta
import threading  # Import threading for background processing
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import glob
import joblib  # For scaler persistence

# Configure logging with rotating file handler
from logging.handlers import RotatingFileHandler
# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('file_mover_app.log', maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVED_MODELS_DIR = "saved_models"

# Ensure the directory exists
if not os.path.exists(SAVED_MODELS_DIR):
    os.makedirs(SAVED_MODELS_DIR)
    logger.info(f"Created directory for saved models at {SAVED_MODELS_DIR}")
# Define the default destination folder (can be changed via UI)
DEFAULT_DESTINATION_FOLDER = os.path.expanduser('~')  # User's home directory as default
class KalmanFilter:
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        if F is None or H is None:
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.m) if R is None else R  # Corrected to use m for R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

        # Precompute identity matrix
        self.I = np.eye(self.n)

    def predict(self, u=0):
        self.x = self.F @ self.x + self.B * u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        y = z - self.H @ self.x
        S = self.R + self.H @ self.P @ self.H.T  # S is scalar
        K = (self.P @ self.H.T) / S  # Equivalent to P H^T (1/S)

        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

    def filterB(self, measurements: np.ndarray):
        num_measurements = measurements.shape[0]
        predictions = np.zeros(num_measurements)

        for i in range(num_measurements):
            self.predict()
            self.update(measurements[i, 0])
            predictions[i] = (self.H @ self.x)[0, 0]

        return predictions

# Define LSTM Autoencoder class
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, num_layers=2, extra_dense_units=128):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder: Stacked LSTM layers
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.relu = nn.ReLU()
        
        # Decoder: Stacked LSTM layers
        self.decoder_lstm_1 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder_lstm_2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        # TimeDistributed Layer: Output is per timestep
        self.time_distributed = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # Encode: Output shape (batch_size, seq_len, hidden_dim)
        x, (h_n, c_n) = self.encoder_lstm(x)
        x = self.relu(x)
        
        # Repeat encoded features to match the sequence length for decoding
        repeat_vector = x[:, -1].unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Decode
        x, _ = self.decoder_lstm_1(repeat_vector)
        x, _ = self.decoder_lstm_2(x)
        
        # Reshape the output for TimeDistributed Layer
        batch_size = x.size(0)
        x = x.contiguous().view(batch_size * self.seq_len, self.hidden_dim)
        
        # TimeDistributed layer for final output
        x = self.time_distributed(x)
        
        # Reshape back to (batch_size, seq_len, input_dim)
        x = x.view(batch_size, self.seq_len, -1)
        
        return x

def train_model(model, train_loader, optimizer, criterion, num_epochs, scaler, model_save_dir=SAVED_MODELS_DIR):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (noisy_batch, clean_batch) in enumerate(train_loader):
            # Move batch to GPU if available
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)

            optimizer.zero_grad()
            # Forward pass
            outputs = model(noisy_batch)
            loss = criterion(outputs, clean_batch)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

        # Check if loss condition is met
        if average_loss <= 0.0167:
            # Create a unique folder name based on loss and timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            folder_name = f"loss_{average_loss:.4f}_time_{timestamp}"
            folder_path = os.path.join(model_save_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"Created folder for model saving at {folder_path}")

            # Save model state_dict
            model_path = os.path.join(folder_path, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")

    # Save the final trained model regardless of loss
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"final_model_time_{timestamp}"
    folder_path = os.path.join(model_save_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    logger.info(f"Created folder for final model saving at {folder_path}")

    model_final_path = os.path.join(folder_path, "final_model.pth")
    torch.save(model.state_dict(), model_final_path)
    logger.info(f"Final model saved to {model_final_path}")

    print(f"Final model saved to {model_final_path}")
    logger.info(f"Final model and scaler saved to {folder_path}")

    # Save the trained model at the end of training
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    print(f"Model saved to {model_path}")
 


# Define function to calculate reconstruction loss (MSE)
def reconstruction_loss(original_data, reconstructed_data):
    return nn.functional.mse_loss(reconstructed_data, original_data, reduction='mean')

# Define function to calculate accuracy based on reconstruction threshold
def calculate_accuracy(test_data, reconstructed_data, threshold=0.01):
    # Calculate the per-sample reconstruction loss
    loss_per_sample = torch.mean((test_data - reconstructed_data) ** 2, dim=[1, 2])  # MSE per sample
    
    # Classify as "correct" if the loss is less than the threshold
    correct = (loss_per_sample < threshold).float()
    
    # Calculate accuracy
    accuracy = correct.mean().item() * 100
    return accuracy

def read_dat_files(folder_path, seq_len, input_dim, smoothing_method=None, window_size=5, apply_kalman=True):
    dat_files = glob.glob(os.path.join(folder_path, '*.dat'))
    all_data = []

    for file in dat_files:
        data = np.fromfile(file, dtype=np.int8)  # 8-bit signed integers
        data = data * -1

        if len(data) < seq_len * input_dim:
            data = np.pad(data, (0, seq_len * input_dim - len(data)), 'constant')
        elif len(data) > seq_len * input_dim:
            data = data[:seq_len * input_dim]

        reshaped_data = data.reshape(seq_len, input_dim)
        all_data.append(reshaped_data)

    return np.array(all_data)

def smooth_data(data, window_size=5, method='moving_average'):
    if len(data) < window_size:
        return data  # If the data is too small, return it unchanged

    if method == 'moving_average':
        cumsum = np.cumsum(np.insert(data, 0, 0)) 
        smoothed_data = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        if len(smoothed_data) > 0:
            smoothed_data = np.pad(smoothed_data, (window_size//2, window_size - window_size//2 - 1), mode='edge')
    
    elif method == 'savgol':
        smoothed_data = savgol_filter(data, window_length=5, polyorder=1)

    return smoothed_data


def kalman_filter_filterpy(data):

    clipped_data = data.copy()
    # Define percentiles
    lower_percentile = 5
    upper_percentile = 95

    # Compute the percentile bounds
    lower_bound = np.percentile(clipped_data, lower_percentile)
    upper_bound = np.percentile(clipped_data, upper_percentile)

    range_band = (lower_bound, upper_bound)

    # Clip the data to the percentile bounds
    clipped_data = np.clip(clipped_data, lower_bound, upper_bound)

    # Update the clipped data
    clipped_data = clipped_data.astype(data.dtype)
    dt = 1000.0 / 7680.0  # Adjusted time step

    # Define the state transition matrix F
    F = np.array([[1, dt],
                  [0, 1]])

    # Define the observation matrix H
    H = np.array([[1, 0]])

    # Define the process noise covariance Q
    Q = np.array([[1, 0],
                  [0, 3]])

    # Define the measurement noise covariance R
    R = np.array([[5]])

    # Define the initial estimation covariance P
    P = np.array([[1000, 0],
                  [0, 1000]])

    # Initialize the Kalman Filter
    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, P=P, x0=np.array([[0], [0]]))

    measurements = clipped_data.reshape(-1, 1)
    kalman_predictions = kf.filterB(measurements)

            # Convert predictions to NumPy array for plotting
    kalman_filtered = np.array(kalman_predictions).astype(np.int8)  # Extract scalar

    # Return a single flattened NumPy array
    return np.array(kalman_filtered.flatten())


def add_noise(data, noise_factor=1):
    noise = noise_factor * np.random.normal(size=data.shape)
    noisy_data = data + noise
    return np.clip(noisy_data, -65, 0) 

# Define function to load the LSTM model
def load_lstm_model(model_path, device, input_dim, hidden_dim, seq_len, num_layers=2, extra_dense_units=128):
    model = LSTMAutoencoder(input_dim, hidden_dim, seq_len, num_layers, extra_dense_units)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"LSTM model loaded from {model_path}")
    return model

# Define function to apply LSTM on data
def apply_lstm(model, data, device):
    """
    Apply the LSTM model to the input data.
    
    Parameters:
    - model: The pre-trained LSTM Autoencoder model.
    - data (np.ndarray): Input data of shape (seq_len, input_dim).
    - device: The device to run the model on.
    
    Returns:
    - np.ndarray: The reconstructed data from the LSTM model.
    """
    with torch.no_grad():
        # Prepare data: add batch dimension and convert to tensor
        input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, seq_len, input_dim)
        reconstructed = model(input_tensor)  # Shape: (1, seq_len, input_dim)
        reconstructed = reconstructed.squeeze(0).cpu().numpy()  # Shape: (seq_len, input_dim)
    return reconstructed

# Define function to save processed data
def save_processed_data(original_filepath: str, processed_data: np.ndarray, suffix: str = '_processed', show_message: bool = True) -> Optional[str]:
    try:
        directory, filename = os.path.split(original_filepath)
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now()  # Use current time for accurate timestamp
        timestamp_str = timestamp.strftime('%Y%m%d%H%M%S%f')  # Includes microseconds

        # Construct the filename with original name and timestamp
        new_filename = f"{name}{suffix}_{timestamp_str}{ext}"
        new_filepath = os.path.join(directory, new_filename)

        # Prevent overwriting existing files
        if os.path.exists(new_filepath):
            overwrite = messagebox.askyesno("Overwrite Confirmation",
                                            f"The file {new_filename} already exists.\nDo you want to overwrite it?")
            if not overwrite:
                logger.info(f"User chose not to overwrite existing file: {new_filename}")
                return None

        processed_data.tofile(new_filepath)
        logger.info(f"Saved processed file: {new_filepath}")

        if show_message:
            messagebox.showinfo("Saved", f"Processed data saved to:\n{new_filepath}")
        return new_filepath  # Return the path of the processed file
    except Exception as e:
        if show_message:
            messagebox.showerror("Error", f"Failed to save processed file {os.path.basename(original_filepath)}:\n{e}")
        logger.error(f"Failed to save processed file {os.path.basename(original_filepath)}: {e}")
        return None


def load_dat_file(filepath: str) -> Optional[np.ndarray]:
    """
    Load a .dat file and return its contents as a NumPy array.

    Parameters:
    - filepath (str): Path to the .dat file.

    Returns:
    - Optional[np.ndarray]: Data array if successful, None otherwise.
    """
    try:
        data = np.fromfile(filepath, dtype=np.int8)
        logger.info(f"Loaded file: {os.path.basename(filepath)}")
        return data
    except Exception as e:
        logger.error(f"Failed to load file {os.path.basename(filepath)}: {e}")
        messagebox.showerror("Error", f"Failed to load file {os.path.basename(filepath)}:\n{e}")
        return None
def display_waveform(data: np.ndarray, title: str = 'Waveform Viewer', y_limits: Optional[tuple] = None, label: str = 'Waveform') -> plt.Figure:
    """
    Create a Matplotlib figure displaying the waveform of the data.

    Parameters:
    - data (np.ndarray): NumPy array of the signal.
    - title (str): Title of the plot.
    - y_limits (Optional[tuple]): Tuple specifying y-axis limits (optional).
    - label (str): Label for the plot.

    Returns:
    - plt.Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    x_axis = np.arange(len(data))
    ax.plot(x_axis, data, 'k', label=label)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    if y_limits:
        ax.set_ylim(y_limits)
    return fig
def move_file(filepath: str, destination_folder: str):
    """
    Move the specified file to the destination folder.

    Parameters:
    - filepath (str): Path to the file to move.
    - destination_folder (str): Destination folder path.
    """
    try:
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            logger.info(f"Created destination folder: {destination_folder}")
        shutil.move(filepath, destination_folder)
        logger.info(f"Moved file {os.path.basename(filepath)} to {destination_folder}")
    except Exception as e:
        logger.error(f"Failed to move file {filepath}: {e}")
        raise e
# Define the FolderTab class
class FolderTab:
    """
    Class to represent each folder tab in the application.
    """
    def __init__(self, parent_notebook: ttk.Notebook, folder_path: str, destination_folder: str, lstm_model: nn.Module, device: torch.device):
        self.parent_notebook = parent_notebook
        self.folder_path = folder_path
        self.destination_folder = destination_folder
        self.files = self.get_dat_files()
        self.processed_filepaths = []
        self.lstm_model = lstm_model
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = device
        self.input_dim = 128
        self.seq_len = 60
        # Initialize UI components for the tab
        self.setup_tab()

    def get_dat_files(self) -> List[str]:
        """
        Retrieve all .dat files in the folder.
        """
        return [
            os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path)
            if f.lower().endswith('.dat')
        ]

    def setup_tab(self):
        """
        Set up the UI components within the tab.
        """
        self.frame = Frame(self.parent_notebook)
        self.parent_notebook.add(self.frame, text=os.path.basename(self.folder_path))
        self.parent_notebook.select(self.frame)

        # Top frame for buttons
        top_frame = Frame(self.frame)
        top_frame.pack(pady=10, fill='x')

        self.process_button = Button(top_frame, text="Process File (Kalman)", command=self.process_file, state="disabled")
        self.process_button.pack(side="left", padx=5)

        self.apply_lstm_button = Button(top_frame, text="Apply LSTM", command=self.apply_lstm_to_file, state="disabled")
        self.apply_lstm_button.pack(side="left", padx=5)

        self.move_button = Button(top_frame, text="Move Processed File", command=self.move_file_action, state="disabled")
        self.move_button.pack(side="left", padx=5)

        self.process_all_button = Button(top_frame, text="Process All", command=self.process_all_files, state="disabled")
        self.process_all_button.pack(side="left", padx=5)

        self.process_all_kalman_button = Button(
            top_frame, 
            text="Process All (Kalman Only)", 
            command=self.process_all_kalman, 
            state="disabled"
        )
        self.process_all_kalman_button.pack(side="left", padx=5)
        self.move_all_button = Button(top_frame, text="Move All Processed Files", command=self.move_all_files, state="disabled")
        self.move_all_button.pack(side="left", padx=5)

        # Label to display destination folder
        self.dest_label = Label(top_frame, text=f"Destination: {self.destination_folder}", fg="blue")
        self.dest_label.pack(side="left", padx=10)

        # Main frame containing sidebar and display area
        main_frame = Frame(self.frame)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left sidebar for listbox
        sidebar_frame = Frame(main_frame)
        sidebar_frame.pack(side="left", fill="y", padx=10, pady=10)

        listbox_label = Label(sidebar_frame, text="Available .dat Files:")
        listbox_label.pack()

        self.listbox = Listbox(sidebar_frame, width=40, selectmode=SINGLE)
        self.listbox.pack(side="left", fill="y")

        self.scrollbar = Scrollbar(sidebar_frame, orient="vertical")
        self.scrollbar.config(command=self.listbox.yview)
        self.scrollbar.pack(side="left", fill="y")
        self.listbox.config(yscrollcommand=self.scrollbar.set)

        self.listbox.bind("<<ListboxSelect>>", self.on_file_select)

        # Populate the listbox with filenames
        self.populate_listbox()

        # Right frame for waveform display with tabs
        self.display_frame = Frame(main_frame)
        self.display_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Create tabs for Original, Kalman Processed, and LSTM Processed waveforms
        self.tabs = ttk.Notebook(self.display_frame)
        self.tabs.pack(fill="both", expand=True)

        self.original_tab = Frame(self.tabs)
        self.kalman_tab = Frame(self.tabs)
        self.lstm_tab = Frame(self.tabs)

        self.tabs.add(self.original_tab, text='Original Waveform')
        self.tabs.add(self.kalman_tab, text='Kalman Filtered Waveform')
        self.tabs.add(self.lstm_tab, text='LSTM Processed Waveform')

        # Containers for Matplotlib figures
        self.original_canvas = None
        self.kalman_canvas = None
        self.lstm_canvas = None

        # Add Progress Bar and Status Label
        progress_frame = Frame(self.frame)
        progress_frame.pack(pady=5, fill='x')

        self.progress = ttk.Progressbar(progress_frame, orient='horizontal', length=400, mode='determinate')
        self.progress.pack(side="left", padx=10)

        self.status_label = ttk.Label(progress_frame, text="Idle")
        self.status_label.pack(side="left", padx=10)

        self.scaler=MinMaxScaler(feature_range=(0, 1))

    def update_status_label(self, message):
        """
        Update the status label within the tab.

        Parameters:
        - message (str): The status message to display.
        """
        self.status_label.config(text=message)

    def populate_listbox(self):
        """
        Populate the listbox with available .dat files.
        """
        self.listbox.delete(0, 'end')
        for file in self.files:
            self.listbox.insert('end', os.path.basename(file))
        logger.info(f"Loaded {len(self.files)} .dat files from: {self.folder_path}")

        # Enable "Process All" button if files are loaded
        if self.files:
            self.process_all_button.config(state="normal")
            self.process_all_kalman_button.config(state="normal")
        else:
            self.process_all_button.config(state="disabled")
            self.process_all_kalman_button.config(state="disabled")

    def process_all_kalman(self):
        """
        Process all loaded files in the folder using Kalman filtering only:
        - For each file, clip the data, apply Kalman Filter, save processed data
        - Update the list of processed files
        - Provide progress indication
        """
        if not self.files:
            messagebox.showwarning("No Files", "No files loaded to process.")
            return

        # Disable buttons to prevent re-entry
        self.process_all_kalman_button.config(state="disabled")
        self.process_all_button.config(state="disabled")
        self.process_button.config(state="disabled")
        self.apply_lstm_button.config(state="disabled")
        self.move_all_button.config(state="disabled")
        self.move_button.config(state="disabled")

        # Reset progress bar and status label
        self.progress['maximum'] = len(self.files)
        self.progress['value'] = 0
        self.status_label.config(text="Starting batch processing (Kalman Only)...")

        # Start processing in a separate thread to keep GUI responsive
        threading.Thread(target=self._process_all_kalman_thread, daemon=True).start()

    def _process_all_kalman_thread(self):
        """
        Threaded function to process all files with Kalman filter only.
        """
        processed_count = 0
        failed_files = []

        for idx, file in enumerate(self.files, start=1):
            try:
                # Load the original data
                data = load_dat_file(file)
                if data is None:
                    failed_files.append(file)
                    continue

                # **Apply Kalman Filtering**
                clipped_data = data.copy()

                # Define percentiles
                lower_percentile = 5
                upper_percentile = 95

                # Compute the percentile bounds
                lower_bound = np.percentile(clipped_data, lower_percentile)
                upper_bound = np.percentile(clipped_data, upper_percentile)

                range_band = (lower_bound, upper_bound)

                # Clip the data to the percentile bounds
                clipped_data = np.clip(clipped_data, lower_bound, upper_bound)

                # Update the clipped data
                self.clipped_data = clipped_data.astype(np.int8)

                # Define Kalman Filter parameters
                dt = 1000.0 / 7680.0  # Adjusted time step

                # Define the state transition matrix F
                F = np.array([[1, dt],
                            [0, 1]])

                # Define the observation matrix H
                H = np.array([[1, 0]])

                # Define the process noise covariance Q
                Q = np.array([[1, 0],
                            [0, 3]])

                # Define the measurement noise covariance R
                R = np.array([[5]])

                # Define the initial estimation covariance P
                P = np.array([[1000, 0],
                            [0, 1000]])

                # Initialize the Kalman Filter
                kf = KalmanFilter(F=F, H=H, Q=Q, R=R, P=P, x0=np.array([[0], [0]]))

                # Apply Kalman Filter to the clipped data
                measurements = clipped_data.reshape(-1, 1)
                kalman_predictions = kf.filterB(measurements)

                # Convert predictions to NumPy array for plotting
                kalman_filtered = np.array(kalman_predictions).astype(np.int8)

                # **Save the Kalman-filtered data**
                kalman_filepath = save_processed_data(
                    file, 
                    kalman_filtered.astype(np.int8),  # Convert back to int8 if necessary
                    suffix='_kalman', 
                    show_message=False  # Suppress individual save messages during batch processing
                )

                if kalman_filepath:
                    self.processed_filepaths.append(kalman_filepath)
                    processed_count += 1

            except Exception as e:
                failed_files.append(file)
                logger.error(f"Failed to process file {file}: {e}")

            finally:
                # Update progress bar and status label using thread-safe method
                self.parent_notebook.after(0, self.update_progress, idx, len(self.files))

        # After processing all files, show a summary message
        summary_message = f"Batch Processing Complete:\nSuccessfully processed {processed_count} files with Kalman filter."
        if failed_files:
            summary_message += f"\nFailed to process {len(failed_files)} files."
            failed_files_str = "\n".join([os.path.basename(f) for f in failed_files])
            summary_message += f"\n\nFailed Files:\n{failed_files_str}"

        self.parent_notebook.after(0, lambda: messagebox.showinfo("Batch Processing Complete", summary_message))

        if processed_count > 0:
            # Enable the "Move All Processed Files" button
            self.parent_notebook.after(0, lambda: self.move_all_button.config(state="normal"))
        if failed_files:
            self.parent_notebook.after(0, lambda: messagebox.showwarning("Some Files Failed", f"Failed to process {len(failed_files)} files."))

        # Reset progress bar and status label
        self.parent_notebook.after(0, self.reset_progress)

        # Re-enable buttons
        self.parent_notebook.after(0, lambda: self.process_all_kalman_button.config(state="normal"))
        self.parent_notebook.after(0, lambda: self.process_all_button.config(state="normal"))
        self.parent_notebook.after(0, lambda: self.process_button.config(state="normal"))


    def on_file_select(self, event):
        """
        Handle the event when a file is selected from the listbox.
        Display the original waveform.
        """
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            selected_file = self.files[index]
            data = load_dat_file(selected_file)
            if data is None:
                return

            # Display Original Waveform
            fig_orig = display_waveform(data, title='Original Waveform')

            # Clear previous figure in original tab
            for widget in self.original_tab.winfo_children():
                widget.destroy()

            # Embed the figure in Tkinter
            self.original_canvas = FigureCanvasTkAgg(fig_orig, master=self.original_tab)
            self.original_canvas.draw()
            self.original_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

            # Add Matplotlib navigation toolbar to original tab
            toolbar_orig = NavigationToolbar2Tk(self.original_canvas, self.original_tab)
            toolbar_orig.update()
            self.original_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

            # Switch to Original tab
            self.tabs.select(self.original_tab)

            # Clear Kalman and LSTM waveform displays
            for widget in self.kalman_tab.winfo_children():
                widget.destroy()
            for widget in self.lstm_tab.winfo_children():
                widget.destroy()

            # Disable the "Apply LSTM" button until processing is done
            self.apply_lstm_button.config(state="disabled")
            self.move_button.config(state="disabled")
            self.move_all_button.config(state="disabled")

            # Enable the "Process File" button
            self.process_button.config(state="normal")

    def process_file(self):
        """
        Process the selected file using Kalman filtering:
        - Clip the signal to the 5th and 95th percentiles
        - Apply Kalman Filter to the clipped signal
        - Display the Kalman-filtered waveform
        - Save the Kalman-filtered data
        """
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a file to process.")
            return

        index = selection[0]
        selected_file = self.files[index]
        data = load_dat_file(selected_file)
        if data is None:
            return

        # Apply Kalman Filtering
        try:
            
            clipped_data = data.copy()

            #Define percentiles
            lower_percentile = 5
            upper_percentile = 95

            #Compute the percentile bounds
            lower_bound = np.percentile(clipped_data, lower_percentile)
            upper_bound = np.percentile(clipped_data, upper_percentile)

            range_band = (lower_bound, upper_bound)

            # Clip the data to the percentile bounds
            clipped_data = np.clip(clipped_data, lower_bound, upper_bound)

            # Update the clipped data
            self.clipped_data = clipped_data.astype(np.int8)

            # Define Kalman Filter parameters (using filterpy)
            dt = 1000.0 / 7680.0  # Adjusted time step

            # Define the state transition matrix F
            F = np.array([[1, dt],
                        [0, 1]])

            # Define the observation matrix H
            H = np.array([[1, 0]])

            # Define the process noise covariance Q
            Q = np.array([[1, 0],
                        [0, 3]])

            # Define the measurement noise covariance R
            R = np.array([[5]])

            # Define the initial estimation covariance P
            P = np.array([[1000, 0],
                        [0, 1000]])

            # Initialize the Kalman Filter
            kf = KalmanFilter(F=F, H=H, Q=Q, R=R, P=P, x0=np.array([[0], [0]]))

            # Apply Kalman Filter to the clipped data
            measurements = clipped_data.reshape(-1, 1)
            kalman_predictions = kf.filterB(measurements)

            # Convert predictions to NumPy array for plotting
            kalman_filtered = np.array(kalman_predictions).astype(np.int8)

            # Display Kalman-filtered waveform
            fig_kalman = display_waveform(kalman_filtered, title='Kalman Filtered Waveform', label='Kalman Filtered')

            # Clear previous figure in Kalman tab
            for widget in self.kalman_tab.winfo_children():
                widget.destroy()

            # Embed the figure in Tkinter
            self.kalman_canvas = FigureCanvasTkAgg(fig_kalman, master=self.kalman_tab)
            self.kalman_canvas.draw()
            self.kalman_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

            # Add Matplotlib navigation toolbar to Kalman tab
            toolbar_kalman = NavigationToolbar2Tk(self.kalman_canvas, self.kalman_tab)
            toolbar_kalman.update()
            self.kalman_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

            # Save the Kalman-filtered data
            kalman_filepath = save_processed_data(
                selected_file, 
                kalman_filtered.astype(np.int8),  # Convert back to int8 if necessary
                suffix='_kalman', 
                show_message=True
            )

            if kalman_filepath:
                self.processed_filepaths.append(kalman_filepath)
                messagebox.showinfo("Processing Complete", "File processed with Kalman filter and saved successfully.")
                # Enable the "Apply LSTM" and "Move" buttons
                self.apply_lstm_button.config(state="normal")
                self.move_button.config(state="normal")
                self.move_all_button.config(state="normal")
                # Switch to Kalman tab to view the waveform
                self.tabs.select(self.kalman_tab)
        except Exception as e:
            logger.error(f"Failed to apply Kalman filter to file {selected_file}: {e}")
            messagebox.showerror("Error", f"Failed to apply Kalman filter to file {os.path.basename(selected_file)}:\n{e}")

    def apply_lstm_to_file(self):
        """
        Apply the LSTM Autoencoder to the last processed (Kalman-filtered) file:
        - Load the Kalman-filtered data
        - Apply LSTM model to reconstruct the data
        - Display the LSTM-processed waveform
        - Save the LSTM-processed data
        """
        if not self.processed_filepaths:
            messagebox.showwarning("No Processed File", "There are no processed files to apply LSTM. Please process files first.")
            return

        # Get the last processed file (Kalman-filtered)
        kalman_filepath = self.processed_filepaths[-1]
        try:
            # Load the Kalman-filtered data
            kalman_data = np.fromfile(kalman_filepath, dtype=np.int8)
            kalman_data = kalman_data * -1  # Reverse the earlier multiplication if necessary

            # Ensure data length matches (seq_len * input_dim)
            if len(kalman_data) != self.seq_len * self.input_dim:
                messagebox.showerror("Error", f"Data size mismatch for file {os.path.basename(kalman_filepath)}.")
                return

            # Reshape data to (seq_len, input_dim)
            kalman_data = kalman_data.reshape(self.seq_len, self.input_dim)

            # Scale data using the pre-fitted scaler
            scaled_kalman_data = self.scaler.fit_transform(kalman_data.reshape(-1, 1)).flatten().reshape(self.seq_len, self.input_dim)

            # Convert to tensor
            scaled_kalman_tensor = torch.tensor(scaled_kalman_data, dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1, seq_len, input_dim)

            # Apply LSTM model
            lstm_reconstructed = self.lstm_model(scaled_kalman_tensor).cpu().detach().numpy().squeeze(0)  # Shape: (seq_len, input_dim)

            # Inverse scale the data
            lstm_reconstructed = self.scaler.inverse_transform(lstm_reconstructed.reshape(-1, 1)).flatten()

            # Convert back to int8 and reverse multiplication if necessary
            lstm_processed = (lstm_reconstructed * -1).astype(np.int8)

            # Display LSTM-processed waveform
            fig_lstm = display_waveform(lstm_processed, title='LSTM Processed Waveform', label='LSTM Processed')

            # Clear previous figure in LSTM tab
            for widget in self.lstm_tab.winfo_children():
                widget.destroy()

            # Embed the figure in Tkinter
            self.lstm_canvas = FigureCanvasTkAgg(fig_lstm, master=self.lstm_tab)
            self.lstm_canvas.draw()
            self.lstm_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

            # Add Matplotlib navigation toolbar to LSTM tab
            toolbar_lstm = NavigationToolbar2Tk(self.lstm_canvas, self.lstm_tab)
            toolbar_lstm.update()
            self.lstm_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

            # Save the LSTM-processed data
            lstm_filepath = save_processed_data(
                kalman_filepath, 
                lstm_processed, 
                suffix='_lstm', 
                show_message=True
            )

            if lstm_filepath:
                self.processed_filepaths.append(lstm_filepath)
                messagebox.showinfo("LSTM Processing Complete", "File processed with LSTM and saved successfully.")
                # Switch to LSTM tab to view the waveform
                self.tabs.select(self.lstm_tab)
        except Exception as e:
            logger.error(f"Failed to apply LSTM to file {kalman_filepath}: {e}")
            messagebox.showerror("Error", f"Failed to apply LSTM to file {os.path.basename(kalman_filepath)}:\n{e}")
    def process_all_files(self):
        """
        Process all loaded files in the folder using Kalman filtering:
        - For each file, clip the data, apply Kalman Filter, save processed data
        - Update the list of processed files
        - Provide progress indication
        """
        if not self.files:
            messagebox.showwarning("No Files", "No files loaded to process.")
            return

        # Disable buttons to prevent re-entry
        self.process_all_button.config(state="disabled")
        self.process_button.config(state="disabled")
        self.apply_lstm_button.config(state="disabled")
        self.move_all_button.config(state="disabled")
        self.move_button.config(state="disabled")

        # Reset progress bar and status label
        self.progress['maximum'] = len(self.files)
        self.progress['value'] = 0
        self.status_label.config(text="Starting batch processing...")

        # Start processing in a separate thread to keep GUI responsive
        threading.Thread(target=self._process_all_files_thread, daemon=True).start()

    def _process_all_files_thread(self):
        processed_count = 0
        failed_files = []

        for idx, file in enumerate(self.files, start=1):
            try:
                # Load the original data
                data = load_dat_file(file)
                if data is None:
                    failed_files.append(file)
                    continue

                # Apply Kalman Filtering
                kalman_filtered = kalman_filter_filterpy(data).astype(np.int8)  # Returns a NumPy array
                kalman_filtered = kalman_filtered * -1  # Reverse the earlier multiplication if necessary

                # Apply LSTM Filter
                # Reshape data to (seq_len, input_dim)
                if len(kalman_filtered) != self.seq_len * self.input_dim:
                    # Adjust the length if necessary
                    if len(kalman_filtered) < self.seq_len * self.input_dim:
                        kalman_filtered = np.pad(kalman_filtered, (0, self.seq_len * self.input_dim - len(kalman_filtered)), 'constant')
                    else:
                        kalman_filtered = kalman_filtered[:self.seq_len * self.input_dim]

                kalman_data = kalman_filtered.reshape(self.seq_len, self.input_dim)

                # Scale data using the scaler
                scaled_kalman_data = self.scaler.fit_transform(kalman_data.reshape(-1, 1)).flatten().reshape(self.seq_len, self.input_dim)

                # Convert to tensor
                scaled_kalman_tensor = torch.tensor(scaled_kalman_data, dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1, seq_len, input_dim)

                # Apply LSTM model
                lstm_reconstructed = self.lstm_model(scaled_kalman_tensor).cpu().detach().numpy().squeeze(0)  # Shape: (seq_len, input_dim)

                # Inverse scale the data
                lstm_reconstructed = self.scaler.inverse_transform(lstm_reconstructed.reshape(-1, 1)).flatten()

                # Convert back to int8 and reverse multiplication if necessary
                lstm_processed = (lstm_reconstructed * -1).astype(np.int8).reshape(self.seq_len * self.input_dim)

                # Save the LSTM-processed data
                lstm_filepath = save_processed_data(
                    file, 
                    lstm_processed, 
                    suffix='_kalman_lstm',  # Use a combined suffix for clarity
                    show_message=False  # Suppress individual save messages during batch processing
                )

                if lstm_filepath:
                    self.processed_filepaths.append(lstm_filepath)
                    processed_count += 1

            except Exception as e:
                failed_files.append(file)
                logger.error(f"Failed to process file {file}: {e}")

            finally:
                # Update progress bar and status label using thread-safe method
                self.parent_notebook.after(0, self.update_progress, idx, len(self.files))

        # After processing all files, show a summary message
        summary_message = f"Batch Processing Complete:\nSuccessfully processed {processed_count} files."
        if failed_files:
            summary_message += f"\nFailed to process {len(failed_files)} files."
            failed_files_str = "\n".join([os.path.basename(f) for f in failed_files])
            summary_message += f"\n\nFailed Files:\n{failed_files_str}"

        self.parent_notebook.after(0, lambda: messagebox.showinfo("Batch Processing Complete", summary_message))

        if processed_count > 0:
            # Enable the "Move All Processed Files" button
            self.parent_notebook.after(0, lambda: self.move_all_button.config(state="normal"))
        if failed_files:
            self.parent_notebook.after(0, lambda: messagebox.showwarning("Some Files Failed", f"Failed to process {len(failed_files)} files."))

        # Reset progress bar and status label
        self.parent_notebook.after(0, self.reset_progress)

        # Re-enable buttons
        self.parent_notebook.after(0, lambda: self.process_all_button.config(state="normal"))
        self.parent_notebook.after(0, lambda: self.process_button.config(state="normal"))

    def update_progress(self, current: int, total: int):
        """
        Update the progress bar and status label.
        """
        self.progress['value'] = current
        self.status_label.config(text=f"Processed {current} of {total} files.")

    def reset_progress(self):
        """
        Reset the progress bar and status label.
        """
        self.progress['value'] = 0
        self.status_label.config(text="Idle")

    def move_file_action(self):
        """
        Move the last processed file to the destination folder.
        """
        if not self.processed_filepaths:
            messagebox.showwarning("No Processed File", "There are no processed files to move. Please process files first.")
            return

        # Move the last processed file
        filepath = self.processed_filepaths[-1]

        confirm = messagebox.askyesno("Confirm Move", f"Are you sure you want to move '{os.path.basename(filepath)}' to the destination folder?")
        if confirm:
            try:
                move_file(filepath, self.destination_folder)
                # Remove the processed file from tracking
                self.processed_filepaths.pop()
                self.move_button.config(state="disabled")
                # Optionally, remove the processed file from the display
                for widget in self.kalman_tab.winfo_children():
                    widget.destroy()
                for widget in self.lstm_tab.winfo_children():
                    widget.destroy()
                messagebox.showinfo("Move Complete", "Processed file has been moved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to move file:\n{e}")

    def move_all_files(self):
        """
        Move all processed files to the destination folder.
        """
        if not self.processed_filepaths:
            messagebox.showwarning("No Processed Files", "There are no processed files to move.")
            return

        confirm = messagebox.askyesno("Confirm Move All", f"Are you sure you want to move all {len(self.processed_filepaths)} processed files to the destination folder?")
        if confirm:
            moved_files = []
            failed_files = []
            for filepath in self.processed_filepaths[:]:  # Copy the list to avoid modification during iteration
                try:
                    move_file(filepath, self.destination_folder)
                    moved_files.append(filepath)
                    self.processed_filepaths.remove(filepath)
                except Exception as e:
                    failed_files.append(filepath)
                    logger.error(f"Failed to move file {filepath}: {e}")

            # Prepare summary message
            summary_message = ""
            if moved_files:
                summary_message += f"Moved {len(moved_files)} files successfully."
            if failed_files:
                summary_message += f"\nFailed to move {len(failed_files)} files."
                failed_files_str = "\n".join([os.path.basename(f) for f in failed_files])
                summary_message += f"\n\nFailed Files:\n{failed_files_str}"

            messagebox.showinfo("Move All Complete", summary_message)

            # Disable the "Move All Processed Files" button if no files left
            if not self.processed_filepaths:
                self.move_all_button.config(state="disabled")

# Define the main application class
class FileMoverApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Waveform Viewer and File Mover")
        self.destination_folder = DEFAULT_DESTINATION_FOLDER  # Initialize with default
        self.destination_tabs = {}  # To keep track of FolderTab instances

        # Set up the UI components first
        self.setup_ui()

        # Set up the LSTM model
        self.lstm_model = initialize_model_and_scaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")

        # Check if model is loaded successfully
        if self.lstm_model is None:
            messagebox.showwarning("Model Not Loaded", "No pre-trained LSTM model loaded. You can train a new model from scratch.")
            logger.warning("Initialization of LSTM model failed or model not found.")
            # Keep the "Train LSTM" button enabled to allow training from scratch
            self.train_button.config(state="normal")
        else:
            # Enable the "Train LSTM" button
            self.train_button.config(state="normal")

    def setup_ui(self):
        # Top frame for buttons and options
        top_frame = Frame(self.root)
        top_frame.pack(pady=10, fill='x')

        self.load_button = Button(top_frame, text="Load Folder", command=self.load_folder)
        self.load_button.pack(side="left", padx=5)

        self.select_dest_button = Button(top_frame, text="Select Destination", command=self.select_destination)
        self.select_dest_button.pack(side="left", padx=5)

        # Display the current destination folder
        self.dest_label = Label(top_frame, text=f"Destination: {self.destination_folder}", fg="blue")
        self.dest_label.pack(side="left", padx=10)

        # Frame for training options
        train_options_frame = Frame(top_frame)
        train_options_frame.pack(side="left", padx=10)

        # Checkbox for "Train from Scratch"
        self.train_from_scratch_var = tk.BooleanVar()
        self.train_from_scratch_checkbox = ttk.Checkbutton(
            train_options_frame,
            text="Train from Scratch",
            variable=self.train_from_scratch_var
        )
        self.train_from_scratch_checkbox.pack(side="left", padx=5)

        # Label and Entry for Number of Epochs
        self.epochs_label = Label(train_options_frame, text="Epochs:")
        self.epochs_label.pack(side="left", padx=5)

        self.epochs_entry = ttk.Entry(train_options_frame, width=5)
        self.epochs_entry.insert(0, "10")  # Default value
        self.epochs_entry.pack(side="left", padx=5)

        self.train_button = Button(top_frame, text="Train LSTM", command=self.train_lstm, state="disabled")
        self.train_button.pack(side="left", padx=5)

        # Add "Load Model" button
        self.load_model_button = Button(top_frame, text="Load Model", command=self.load_model, state="normal")
        self.load_model_button.pack(side="left", padx=5)

        self.load_pth_model_button = Button(top_frame, text="Load .pth Model", command=self.load_pth_model, state="normal")
        self.load_pth_model_button.pack(side="left", padx=5)

        # Main frame containing the Notebook for tabs
        main_frame = Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        # Notebook to hold multiple folder tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True)

        # Bind right-click on tabs to show context menu
        self.notebook.bind("<Button-3>", self.show_tab_context_menu)

        # Create a context menu
        self.tab_context_menu = Menu(self.root, tearoff=0)
        self.tab_context_menu.add_command(label="Close", command=self.close_selected_tab)
          # Add a dedicated frame for training progress
        training_progress_frame = Frame(self.root)
        training_progress_frame.pack(pady=5, fill='x')

        self.train_progress = ttk.Progressbar(training_progress_frame, orient='horizontal', length=400, mode='determinate')
        self.train_progress.pack(side="left", padx=10)

        self.train_status_label = ttk.Label(training_progress_frame, text="Idle")
        self.train_status_label.pack(side="left", padx=10)


    def load_pth_model(self):
        """
        Handler for the "Load .pth Model" button.
        Allows the user to select a .pth model file via a file picker and loads it into the application.
        """
        # Open a file dialog to select the .pth model file
        model_file_path = filedialog.askopenfilename(
            title="Select .pth Model File",
            filetypes=[("PyTorch Model Files", "*.pth"), ("All Files", "*.*")]
        )

        if not model_file_path:
            logger.info("No model file selected.")
            return  # User canceled the dialog

        # Load the selected model
        try:
            new_model = load_lstm_model(
                model_path=model_file_path,
                device=self.device,
                input_dim=128,
                hidden_dim=256,
                seq_len=60,
                num_layers=1,
                extra_dense_units=64
            )
            self.lstm_model = new_model  # Update the application's model

            # Update all FolderTabs with the new model
            for tab in self.destination_tabs.values():
                tab.lstm_model = new_model

            logger.info(f"Loaded model from {model_file_path}")
            messagebox.showinfo("Model Loaded", f"Model loaded successfully from:\n{model_file_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_file_path}: {e}")
            messagebox.showerror("Model Load Error", f"Failed to load model:\n{e}")


    def load_folder(self):
        """
        Handler for the "Load Folder" button.
        Allows the user to select a folder and creates a new FolderTab for it.
        """
        folder = filedialog.askdirectory(title="Select Folder to Load")
        if folder:
            # Check if the folder is already loaded as a tab
            for tab_id, tab in self.destination_tabs.items():
                if tab.folder_path == folder:
                    messagebox.showinfo("Folder Already Loaded", "The selected folder is already loaded.")
                    return

            # Create a new FolderTab
            new_tab = FolderTab(
                parent_notebook=self.notebook,
                folder_path=folder,
                destination_folder=self.destination_folder,
                lstm_model=self.lstm_model,
                device=self.device
            )

            # Add the new tab to the notebook
            tab_id = self.notebook.tabs()[-1]  # Assuming it's the last tab added
            self.destination_tabs[tab_id] = new_tab
            logger.info(f"Loaded new folder: {folder}")
            messagebox.showinfo("Folder Loaded", f"Folder '{os.path.basename(folder)}' loaded successfully.")
        else:
            logger.info("No folder selected to load.")
    def select_destination(self):
        """
        Allow the user to select a destination folder via the UI.
        """
        folder = filedialog.askdirectory(title="Select Destination Folder")
        if folder:
            self.destination_folder = folder
            self.dest_label.config(text=f"Destination: {self.destination_folder}")
            logger.info(f"Destination folder set to: {self.destination_folder}")

            # Update destination folder in all existing tabs
            for tab in self.destination_tabs.values():
                tab.destination_folder = self.destination_folder
                tab.dest_label.config(text=f"Destination: {self.destination_folder}")

    def show_tab_context_menu(self, event):
        """
        Show context menu on right-clicking a tab.
        """
        # Identify the tab that was right-clicked
        clicked_tab = self.get_tab_id(event.x, event.y)
        if clicked_tab is not None:
            self.notebook.select(clicked_tab)  # Select the tab
            self.tab_context_menu.post(event.x_root, event.y_root)

    def get_tab_id(self, x, y):
        """
        Get the tab id based on the x, y coordinates of the mouse event.
        """
        try:
            # The "@x,y" notation can sometimes be unreliable; use "identify" and "index"
            # to ensure accurate tab identification
            element = self.notebook.identify(x, y)
            if element != "label":
                return None
            tab_index = self.notebook.index("@%d,%d" % (x, y))
            return tab_index
        except Exception as e:
            logger.error(f"Failed to identify tab at ({x}, {y}): {e}")
            return None

    def close_selected_tab(self):
        """
        Close the currently selected tab.
        """
        current_tab = self.notebook.select()
        if not current_tab:
            return

        # Find the FolderTab instance associated with the current tab
        tab_to_remove = self.destination_tabs.get(current_tab, None)

        if tab_to_remove:
            confirm = messagebox.askyesno("Close Tab", f"Are you sure you want to close the folder '{os.path.basename(tab_to_remove.folder_path)}'?")
            if confirm:
                # Remove the tab from notebook
                self.notebook.forget(current_tab)
                # Remove the tab instance from the dictionary
                del self.destination_tabs[current_tab]
                logger.info(f"Closed tab for folder: {tab_to_remove.folder_path}")
        else:
            messagebox.showerror("Error", "Failed to identify the folder for the selected tab.")

    def train_lstm(self):
        """
        Handler for the "Train LSTM" button.
        Allows the user to select noisy and clean data folders, choose training mode, specify epochs, and initiates the training process.
        """
        # Retrieve the training mode
        train_from_scratch = self.train_from_scratch_var.get()

        # Retrieve the number of epochs
        try:
            num_epochs = int(self.epochs_entry.get())
            if num_epochs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid Epochs", "Please enter a valid positive integer for epochs.")
            return
        
        messagebox.showinfo("First Step", "Please select the Noisy Data Folder.")

        # Prompt user to select the noisy data folder
        noisy_folder = filedialog.askdirectory(title="Select Noisy Data Folder")
        if not noisy_folder:
            messagebox.showwarning("No Selection", "No noisy data folder selected. Training canceled.")
            return  # User canceled the dialog

        # Notify the user to select the clean data folder next
        messagebox.showinfo("Next Step", "Please select the corresponding Clean Data Folder.")

        # Prompt user to select the clean data folder
        clean_folder = filedialog.askdirectory(title="Select Clean Data Folder")
        if not clean_folder:
            messagebox.showwarning("No Selection", "No clean data folder selected. Training canceled.")
            return  # User canceled the dialog

        # Optionally, prompt for hyperparameters (learning rate)
        # You can enhance this by adding more input fields if needed

        learning_rate = 0.0001  # Default value; you can also make this user-configurable

        # Disable the train button to prevent multiple clicks
        self.train_button.config(state="disabled")

        # Initialize a new model if training from scratch
        print(train_from_scratch)
        if train_from_scratch:
            confirm = messagebox.askyesno("Confirm Training from Scratch",
                                        "Training from scratch will overwrite the existing model.\nDo you want to proceed?")
            if not confirm:
                self.train_button.config(state="normal")
                return
            # Initialize a fresh model
            self.lstm_model = LSTMAutoencoder(
                input_dim=128,
                hidden_dim=256,
                seq_len=60,
                num_layers=1,
                extra_dense_units=64
            ).to(self.device)
            logger.info("Initialized a new LSTM model for training from scratch.")

        # Start training in a separate thread
        threading.Thread(
            target=self._train_lstm_thread,
            args=(noisy_folder, clean_folder, num_epochs, learning_rate, train_from_scratch),
            daemon=True
        ).start()

        # self.lstm_model =   # Replace with the actual trained model instance

        # Update all FolderTabs with the new model
        for tab in self.destination_tabs.values():
            tab.lstm_model = self.lstm_model

    def load_model(self):
        """
        Open a dialog to list saved models and allow the user to select one to load.
        """
        # Get all saved model folders
        model_folders = sorted(
            glob.glob(os.path.join(SAVED_MODELS_DIR, "loss_*")),
            key=os.path.getmtime,
            reverse=True
        ) + sorted(
            glob.glob(os.path.join(SAVED_MODELS_DIR, "final_model_*")),
            key=os.path.getmtime,
            reverse=True
        )

        if not model_folders:
            messagebox.showinfo("No Models Found", "No saved models are available to load.")
            return

        # Create a list of model names for display
        model_names = [os.path.basename(folder) for folder in model_folders]

        # Create a new window for model selection
        load_window = Tk()
        load_window.title("Load Saved Model")
        load_window.geometry("400x300")

        # Label
        label = Label(load_window, text="Select a model to load:", font=("Helvetica", 14))
        label.pack(pady=10)

        # Listbox with scrollbar
        listbox_frame = Frame(load_window)
        listbox_frame.pack(fill="both", expand=True, padx=20, pady=10)

        scrollbar = Scrollbar(listbox_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        listbox = Listbox(listbox_frame, selectmode=SINGLE, yscrollcommand=scrollbar.set)
        for name in model_names:
            listbox.insert("end", name)
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)

        # Define the on_load function inside load_model to access load_window
        def on_load():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a model to load.")
                return
            index = selection[0]
            selected_folder = model_folders[index]
            model_path = glob.glob(os.path.join(selected_folder, "*.pth"))[0]  # Assuming one .pth per folder

            # Load the model
            try:
                new_model = load_lstm_model(
                    model_path=model_path,
                    device=self.device,
                    input_dim=128,
                    hidden_dim=256,
                    seq_len=60,
                    num_layers=1,
                    extra_dense_units=64
                )
                self.lstm_model = new_model  # Update the application's model

                # Update all FolderTabs with the new model
                for tab in self.destination_tabs.values():
                    tab.lstm_model = new_model

                logger.info(f"Model loaded from {model_path}")
                messagebox.showinfo("Model Loaded", f"Model '{model_names[index]}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                messagebox.showerror("Model Load Error", f"Failed to load model:\n{e}")
                return

            load_window.destroy()

        # Load button
        load_button = Button(load_window, text="Load Selected Model", command=on_load)
        load_button.pack(pady=10)

        # Close button
        close_button = Button(load_window, text="Cancel", command=load_window.destroy)
        close_button.pack(pady=5)

        load_window.mainloop()

    def _train_lstm_thread(self, noisy_folder, clean_folder, num_epochs, learning_rate, train_from_scratch):
        """
        Threaded function to train the LSTM model using noisy-clean data pairs.
        """
        try:
            # Update status
            self.update_status("Training started...")
        
            # Prepare the training data
            train_loader, scaler = self.prepare_training_data(noisy_folder, clean_folder)
        
            if not train_loader:
                self.update_status("No training data found or mismatch in folders.")
                self.enable_train_button()
                return
        
            # Define loss and optimizer
            criterion = nn.MSELoss()  # Reconstruction task with mean squared error
            optimizer = optim.Adam(self.lstm_model.parameters(), lr=learning_rate)
        
            # Set the progress bar maximum
            total_batches = len(train_loader) * num_epochs
            self.root.after(0, lambda: self.train_progress.config(maximum=total_batches))
            self.root.after(0, lambda: self.train_progress.config(value=0))
            current_progress = 0
        
            # Train the model
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for batch_idx, (noisy_batch, clean_batch) in enumerate(train_loader):
                    # Move batch to GPU if available
                    noisy_batch = noisy_batch.to(self.device)
                    clean_batch = clean_batch.to(self.device)
        
                    optimizer.zero_grad()
                    # Forward pass
                    outputs = self.lstm_model(noisy_batch)
                    loss = criterion(outputs, clean_batch)
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
        
                    epoch_loss += loss.item()
                    current_progress += 1
        
                    # Update progress bar
                    self.root.after(0, lambda cp=current_progress: self.train_progress.config(value=cp))
        
                average_loss = epoch_loss / len(train_loader)
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')
        
                # Update status with current loss after epoch
                self.update_status(f"Epoch {epoch+1}/{num_epochs} completed. Loss: {average_loss:.4f}")
        
            # Save the trained model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            folder_name = f"trained_model_{timestamp}"
            folder_path = os.path.join(SAVED_MODELS_DIR, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"Created folder for model saving at {folder_path}")
        
            model_path = os.path.join(folder_path, "trained_model.pth")
            torch.save(self.lstm_model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
        
            # Optionally, export the model to ONNX
            onnx_model_path = os.path.join(folder_path, "trained_model.onnx")
            self.export_model_to_onnx(self.lstm_model, onnx_model_path, folder_path)
        
            # Update status
            self.update_status("Training complete.", progress=total_batches)
        
            # Notify the user
            messagebox.showinfo(
                "Training Complete",
                f"Model trained and saved to:\n{model_path}\nONNX model saved to:\n{onnx_model_path}"
            )
        
            # Update all FolderTabs with the new model
            for tab in self.destination_tabs.values():
                tab.lstm_model = self.lstm_model
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            messagebox.showerror("Training Error", f"An error occurred during training:\n{e}")
            self.update_status("Training failed.")
        finally:
            # Re-enable the train button
            self.enable_train_button()


    def prepare_training_data(self, noisy_folder, clean_folder):
        """
        Prepares the training data, fits the scaler, and returns a DataLoader.
        Assumes that the training data consists of pairs of noisy and clean data.
        
        Parameters:
        - noisy_folder (str): Path to the noisy data folder.
        - clean_folder (str): Path to the clean data folder.
        
        Returns:
        - Tuple[DataLoader, MinMaxScaler]: PyTorch DataLoader for training and the fitted scaler.
        """
        # Retrieve .dat files from both folders
        noisy_files = sorted([
            os.path.join(noisy_folder, f) for f in os.listdir(noisy_folder)
            if f.lower().endswith('.dat')
        ])
        clean_files = sorted([
            os.path.join(clean_folder, f) for f in os.listdir(clean_folder)
            if f.lower().endswith('.dat')
        ])

        if not noisy_files or not clean_files:
            messagebox.showwarning("No Data", "No .dat files found in one or both of the selected folders.")
            return None, None

        # Ensure both folders have the same number of files
        if len(noisy_files) != len(clean_files):
            messagebox.showwarning(
                "Mismatch in Files",
                f"Noisy folder has {len(noisy_files)} files while clean folder has {len(clean_files)} files."
            )
            return None, None

        # Optionally, ensure that filenames correspond
        # For example, 'sample1.dat' in noisy folder corresponds to 'sample1.dat' in clean folder
        paired_files = []
        for noisy_file, clean_file in zip(noisy_files, clean_files):
            noisy_name = os.path.basename(noisy_file)
            clean_name = os.path.basename(clean_file)
            # if noisy_name != clean_name:
            #     messagebox.showwarning(
            #         "Filename Mismatch",
            #         f"Filename mismatch: '{noisy_name}' and '{clean_name}' do not match."
            #     )
            #     return None, None
            paired_files.append((noisy_file, clean_file))

        # Load all clean data to fit the scaler
        all_clean_data = []
        for _, clean_file in paired_files:
            clean_data = load_dat_file(clean_file)
            if clean_data is not None:
                clean_data = clean_data.astype(np.float32)
                all_clean_data.append(clean_data)

        if not all_clean_data:
            messagebox.showwarning("No Valid Data", "No valid clean data found in the selected clean folder.")
            return None, None

        all_clean_data = np.vstack(all_clean_data).reshape(-1, 1)  # Reshape for scaler

        # Initialize and fit the scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(all_clean_data)
        logger.info("Scaler fitted on clean training data.")

        # Create a custom Dataset with the fitted scaler
        class PairedWaveformDataset(torch.utils.data.Dataset):
            def __init__(self, paired_file_list, scaler: MinMaxScaler, noise_factor=0.5):
                self.paired_file_list = paired_file_list
                self.noise_factor = noise_factor
                self.scaler = scaler  # Pre-fitted scaler

            def __len__(self):
                return len(self.paired_file_list)

            def __getitem__(self, idx):
                noisy_file, clean_file = self.paired_file_list[idx]

                # Load noisy and clean data
                noisy_data = load_dat_file(noisy_file)
                clean_data = load_dat_file(clean_file)

                if noisy_data is None or clean_data is None:
                    # If any data is missing, return zeros
                    noisy_data = np.zeros(60 * 128, dtype=np.int8)
                    clean_data = np.zeros(60 * 128, dtype=np.int8)
                noisy_data = noisy_data.astype(np.float32)
                clean_data = clean_data.astype(np.float32)

                # Normalize the clean data using the pre-fitted scaler
                clean_data = self.scaler.transform(clean_data.reshape(-1, 1)).flatten()

                # Optionally, you can add noise to the clean data if noisy data isn't provided
                # But since we have separate noisy data, we can use it directly
                # Ensure noisy data is scaled similarly
                noisy_data = self.scaler.transform(noisy_data.reshape(-1, 1)).flatten()

                # Convert to tensors
                clean_tensor = torch.tensor(clean_data, dtype=torch.float32).reshape(60, 128)
                noisy_tensor = torch.tensor(noisy_data, dtype=torch.float32).reshape(60, 128)
                return noisy_tensor, clean_tensor

        dataset = PairedWaveformDataset(paired_files, scaler=scaler)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        return train_loader, scaler


    def export_model_to_onnx(self, model, onnx_path, folder_path):
        """
        Exports the PyTorch model to ONNX format.

        Parameters:
        - model (nn.Module): The trained PyTorch model.
        - onnx_path (str): Path to save the ONNX model.
        - folder_path (str): Path to the folder where the model is saved.
        """
        try:
            model.eval()
            dummy_input = torch.randn(1, 60, 128).to(self.device)  # Adjust the input size as per your model
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                opset_version=11
            )
            logger.info(f"Model exported to ONNX format at {onnx_path}")
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
            messagebox.showwarning("ONNX Export Failed", f"Failed to export model to ONNX:\n{e}")

    def update_status(self, message, progress=None):
        """
        Update the global training status label and progress bar.

        Parameters:
        - message (str): The status message to display.
        - progress (int, optional): The progress value to set on the progress bar.
        """
        self.root.after(0, lambda: self.train_status_label.config(text=message))
        if progress is not None:
            self.root.after(0, lambda: self.train_progress.config(value=progress))
    
    # def update_status_label(self, message):
    #     """
    #     Update the status label (to be called from the main thread).

    #     Parameters:
    #     - message (str): The status message to display.
    #     """
    #     # Find the progress_frame and update the status_label
    #     for child in self.root.winfo_children():
    #         if isinstance(child, Frame):
    #             for subchild in child.winfo_children():
    #                 if isinstance(subchild, Label) and subchild.cget("text").startswith("Destination:"):
    #                     # Assuming the status_label is defined separately
    #                     pass  # Modify as per your actual status_label path
    #     # Alternatively, you can store a reference to the status_label
    #     # For simplicity, let's assume each FolderTab has its own status_label
    #     # So, you may need to iterate through tabs and update their status_labels
    #     for tab in self.destination_tabs.values():
    #         tab.update_status_label(message)

    def enable_train_button(self):
        """
        Re-enable the "Train LSTM" button.
        """
        self.root.after(0, lambda: self.train_button.config(state="normal"))

# Define function to load the LSTM model and scaler
def initialize_model_and_scaler():
    """
    Initialize the LSTM model.
    Returns the model if successful, else None.
    """
    model_path = "lstm_autoencoder_model_23.pth"  # Ensure this file exists in the project root
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_path):
        messagebox.showerror("Model Not Found", f"LSTM model file '{model_path}' not found in the project root.")
        logger.error(f"LSTM model file '{model_path}' not found.")
        return None

    # Load the LSTM model
    model = load_lstm_model(model_path, device, input_dim=128, hidden_dim=256, seq_len=60, num_layers=1, extra_dense_units=64)

    return model



if __name__ == "__main__":
    root = Tk()
    root.geometry("1300x800")  # Increased size for better visualization
    app = FileMoverApp(root)
    root.mainloop()
