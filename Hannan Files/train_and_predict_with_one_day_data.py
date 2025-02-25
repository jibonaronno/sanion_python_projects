import os
import glob
import numpy as np
import pywt
from scipy.signal import convolve, hilbert
import matplotlib.pyplot as plt

# -------------------------------
# PyTorch Imports
# -------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Denoising and Smoothing Functions
# -------------------------------
def wavelet_denoise(signal, wavelet_name='db4', level=None, thresholding='hard', threshold_multiplier=2.0):
    """
    Apply wavelet denoising to the signal.
    """
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745  # Robust estimator of standard deviation.
    uthresh = threshold_multiplier * sigma * np.sqrt(2 * np.log(len(signal)))
    
    denoised_coeffs = [coeffs[0]]  # Keep the approximation coefficients unchanged.
    for detail in coeffs[1:]:
        denoised_detail = pywt.threshold(detail, value=uthresh, mode=thresholding)
        denoised_coeffs.append(denoised_detail)
    
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet_name)
    return denoised_signal[:len(signal)]

def gaussian_kernel(kernel_size=5, sigma=1.0):
    """
    Create a normalized 1D Gaussian kernel.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure kernel_size is odd.
    half_size = kernel_size // 2
    x = np.arange(-half_size, half_size + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def apply_conv_filter(signal, kernel):
    """
    Apply a one-dimensional convolution filter to the signal.
    """
    return convolve(signal, kernel, mode='same')

# -------------------------------
# Process Files in Batches (Batch = One Minute)
# -------------------------------
# Specify the folder containing your .dat files.
folder_path = './combined2'  # <-- UPDATE with your folder path.
file_pattern = os.path.join(folder_path, '*.dat')
file_list = sorted(glob.glob(file_pattern))

# Processing parameters
sampling_rate = 7680         # samples per second (each file is 1 second long)
window_size = 100            # Window size for moving average (trend extraction)
files_per_minute = 60        # Since each file is 1 second, 60 files = 1 minute

# We will store one composite anomaly score per minute
anomaly_score_minute_list = []
time_minutes = []
minute_counter = 0
start_collect = 0
end_collect = 5000

print("Processing files in batches (per minute)...")
for i in range(start_collect, end_collect, files_per_minute):
    if end_collect - i < files_per_minute:
        break
    batch_files = file_list[i:i+files_per_minute]
    signals_list = []
    trends_list = []
    
    for file in batch_files:
        # Load data from file (each file is assumed to contain 7680 1-byte signed integers).
        data = np.fromfile(file, dtype=np.int8)
        if data.size == 0:
            print(f"Skipping empty file: {file}")
            continue
        
        # Step 1: Noise Reduction using wavelet denoising.
        denoised_signal = wavelet_denoise(data, wavelet_name='db4', level=None,
                                          thresholding='hard', threshold_multiplier=2.0)
        kernel = gaussian_kernel(kernel_size=5, sigma=1.0)
        smoothed_signal = apply_conv_filter(denoised_signal, kernel)
        signals_list.append(smoothed_signal)
        
        # Step 2: Trend Extraction using a simple moving average.
        trend = np.convolve(smoothed_signal, np.ones(window_size) / window_size, mode='same')
        trends_list.append(trend)
    
    # If no files in this batch produced valid data, skip it.
    if len(signals_list) == 0:
        continue
    
    # Combine the processed signals and trends from this batch (minute).
    combined_signal = np.concatenate(signals_list)
    combined_trend = np.concatenate(trends_list)
    
    # -------------------------------
    # Compute Composite Anomaly Score (Per Minute)
    # -------------------------------
    # (A) Compute the residual: difference between the smoothed signal and its trend.
    residual = combined_signal - combined_trend
    
    # (B) Compute the amplitude envelope using the Hilbert transform.
    analytic_signal = hilbert(combined_signal)
    amplitude_envelope = np.abs(analytic_signal)
    
    # (C) Normalize the residual and amplitude envelope (z-scores computed on this minute's data).
    residual_z = (residual - np.mean(residual)) / np.std(residual)
    amplitude_z = (amplitude_envelope - np.mean(amplitude_envelope)) / np.std(amplitude_envelope)
    
    # (D) Form a composite anomaly score by averaging the absolute deviations.
    anomaly_score = (np.abs(residual_z) + np.abs(amplitude_z)) / 2
    
    # Aggregate the anomaly score over this minute (by taking the mean).
    minute_score = np.mean(anomaly_score)
    anomaly_score_minute_list.append(minute_score)
    time_minutes.append(minute_counter)
    minute_counter += 1

# Convert to numpy arrays
anomaly_score_minute = np.array(anomaly_score_minute_list)
time_minutes = np.array(time_minutes)

print(f"Processed {len(anomaly_score_minute)} minutes of data.")

# -------------------------------
# Create Sliding-Window Dataset for CNN Forecasting (Per Minute)
# -------------------------------
# Each input sequence will consist of anomaly scores for 'seq_length' minutes
# and the target will be the anomaly score for the next minute.
seq_length = 10
X_seq = []
y_seq = []
for i in range(len(anomaly_score_minute) - seq_length):
    X_seq.append(anomaly_score_minute[i:i+seq_length])
    y_seq.append(anomaly_score_minute[i+seq_length])
    
X_seq = np.array(X_seq)  # shape: (num_samples, seq_length)
y_seq = np.array(y_seq)  # shape: (num_samples,)

# Reshape X_seq for PyTorch Conv1d: (batch_size, channels, sequence_length)
X_seq = X_seq.reshape(-1, 1, seq_length)

# -------------------------------
# Split Dataset into Training and Test Sets (Per Minute)
# -------------------------------
train_ratio = 0.8
num_samples = X_seq.shape[0]
train_size = int(num_samples * train_ratio)

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]
X_test = X_seq[train_size:]
y_test = y_seq[train_size:]

# Convert to PyTorch tensors.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# -------------------------------
# Define a Convolutional Neural Network Model for Time Series Forecasting (Per Minute)
# -------------------------------
class ConvTimeSeriesNN(nn.Module):
    def __init__(self, seq_length):
        super(ConvTimeSeriesNN, self).__init__()
        # Input shape: (batch_size, 1, seq_length)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # output: (batch, 16, seq_length)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)  # output: (batch, 16, seq_length//2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # output: (batch, 32, seq_length//2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # output: (batch, 32, seq_length//4)
        
        # Compute the size after convolution and pooling
        self.feature_size = (seq_length // 4) * 32
        
        self.fc1 = nn.Linear(self.feature_size, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.conv1(x)   # shape: (batch, 16, seq_length)
        x = self.relu(x)
        x = self.pool(x)    # shape: (batch, 16, seq_length//2)
        x = self.conv2(x)   # shape: (batch, 32, seq_length//2)
        x = self.relu(x)
        x = self.pool2(x)   # shape: (batch, 32, seq_length//4)
        x = x.view(-1, self.feature_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model and move it to the GPU (if available).
model_cnn = ConvTimeSeriesNN(seq_length=seq_length).to(device)

# Define loss function and optimizer.
criterion = nn.MSELoss()
optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)

# -------------------------------
# Train the CNN Model (Per Minute)
# -------------------------------
num_epochs = 300
for epoch in range(num_epochs):
    model_cnn.train()
    optimizer.zero_grad()
    outputs = model_cnn(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model_save_path = "./model_cnn.pt"
torch.save(model_cnn.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

model_cnn_2 = ConvTimeSeriesNN(seq_length=seq_length).to(device)
model_cnn_2.load_state_dict(torch.load("./model_cnn.pt"))
# -------------------------------
# Evaluate the Model on Training and Test Sets (Per Minute)
# -------------------------------
model_cnn_2.eval()
with torch.no_grad():
    y_train_pred = model_cnn(X_train_tensor).cpu().numpy().flatten()
    y_test_pred = model_cnn(X_test_tensor).cpu().numpy().flatten()

# -------------------------------
# Future Predictions (Iterative Forecasting, Per Minute)
# -------------------------------
# We will use the last available window from the entire dataset to forecast the next minutes.
future_steps = 100
last_window = anomaly_score_minute[-seq_length:].copy()  # shape: (seq_length,)
future_predictions = []

model_cnn_2.eval()
with torch.no_grad():
    current_window = last_window.copy()
    for i in range(future_steps):
        # Prepare current window: shape (1, 1, seq_length)
        current_input = torch.tensor(current_window.reshape(1, 1, seq_length), dtype=torch.float32).to(device)
        pred = model_cnn(current_input).item()
        future_predictions.append(pred)
        # Update the window: drop the first value and append the new prediction.
        current_window = np.roll(current_window, -1)
        current_window[-1] = pred

future_time = np.arange(time_minutes[-1] + 1, time_minutes[-1] + future_steps + 1)

# Define desired output range (for display only).
desired_min = -55
desired_max = 0

# We use the min and max of the training target anomaly score (unscaled) for the mapping.
a_min = anomaly_score_minute.min()
a_max = anomaly_score_minute.max()

def scale_values(values, a_min, a_max, desired_min, desired_max):
    """Linearly map values from [a_min, a_max] to [desired_min, desired_max]."""
    return (values - a_min) / (a_max - a_min) * (desired_max - desired_min) + desired_min

anomaly_score_minute_scaled = scale_values(anomaly_score_minute, a_min, a_max, desired_min, desired_max)
y_test_pred_scaled = scale_values(y_test_pred, a_min, a_max, desired_min, desired_max)
future_predictions_scaled = scale_values(np.array(future_predictions), a_min, a_max, desired_min, desired_max)

# -------------------------------
# Visualization (Per Minute)
# -------------------------------
plt.figure(figsize=(15, 6))
# Plot the actual per-minute anomaly scores (scaled for display).
plt.plot(time_minutes, anomaly_score_minute_scaled, label='Actual Anomaly Score (per Minute)', color='blue', marker='o')
#plt.plot(time_minutes, anomaly_score_minute, label='Actual Anomaly Score (per Minute)', color='blue', marker='o')
# For visualization, plot the test predictions (scaled).
# Adjust the time axis for test predictions:
test_time = time_minutes[seq_length + train_size : seq_length + train_size + len(y_test_pred_scaled)]
plt.plot(test_time, y_test_pred_scaled, label='Test Predictions (CNN)', color='orange', linestyle='--', marker='x')
#plt.plot(test_time, y_test_pred, label='Test Predictions (CNN)', color='orange', linestyle='--', marker='x')
# Plot the future predictions (scaled).
plt.plot(future_time, future_predictions_scaled, label='Future Predictions (CNN)', color='magenta', linestyle='--', marker='x')
#plt.plot(future_time, future_predictions, label='Future Predictions (CNN)', color='magenta', linestyle='--', marker='x')
# (Optional) If you have an alarm threshold in the desired scale, plot it.
alarm_threshold = -40  # Change as needed (this threshold is now in the scaled domain)
plt.axhline(y=alarm_threshold, color='red', linestyle='--', label='Alarm Threshold')
plt.title('Composite Anomaly Score and Future Predictions (Per Minute)')
plt.xlabel('Time (minutes)')
plt.ylabel('Anomaly Score (Scaled)')
plt.legend()
plt.tight_layout()
plt.show()
