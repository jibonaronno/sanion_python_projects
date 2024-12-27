import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import glob
import time
import onnxruntime as ort  # ONNX Runtime
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
from scipy.signal import savgol_filter
from pykalman import KalmanFilter
from filterpy.kalman import KalmanFilter
print(torch.__version__)

print(torch.backends.cudnn.version())

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"

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
        
        # # # Add additional fully connected layers and ReLU for complexity
        # self.fc1 = nn.Linear(hidden_dim, extra_dense_units)
        # self.relu_fc1 = nn.ReLU()

        # self.fc2 = nn.Linear(extra_dense_units, hidden_dim)  # Return to hidden_dim for LSTM processing
        # self.relu_fc2 = nn.ReLU()
        
        # Decoder: Stacked LSTM layers
        self.decoder_lstm_1 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder_lstm_2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        # TimeDistributed Layer: Output is per timestep
        self.time_distributed = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # Encode: Output shape (batch_size, seq_len, hidden_dim)
        x, (h_n, c_n) = self.encoder_lstm(x)
        x = self.relu(x)
        
        # # # Add complexity: pass through fully connected layers and ReLU activations
        # x = self.fc1(x)
        # x = self.relu_fc1(x)

        # x = self.fc2(x)
        # x = self.relu_fc2(x)
        
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


def chuyen_doi_nhi_phan_sang_so_am(nhi_phan: str) -> int:
    # Kiểm tra nếu số nhị phân rỗng
    if len(nhi_phan) == 0:
        return 0
    if len(nhi_phan) != 8:
        return 0
    # Kiểm tra bit dấu
    dau = int(nhi_phan[0])
    # Áp dụng phương pháp bù hai nếu số nhị phân là số âm
    if dau == 1:
        # Đảo bit
        dao_bit = "".join("1" if bit == "0" else "0" for bit in nhi_phan[1:])
        # Chuyển đổi sang số nguyên dương ban đầu
        so_nguyen_duong = int(dao_bit, 2)
        # Lấy bù 1 của số nguyên dương
        bu_1 = ~so_nguyen_duong
        # Chuyển đổi thành số âm
        so_am = bu_1 - 1
        return so_am
    else:
        # Số nhị phân là số dương
        so_nguyen_duong = int(nhi_phan, 2)
        return so_nguyen_duong
    

def extraProcessingFile(x):
    _data = bin(x)[2:]
    return chuyen_doi_nhi_phan_sang_so_am(_data)


def read_dat_files(folder_path, seq_len, input_dim, smoothing_method=None, window_size=5, apply_kalman=True):
    # Get all .dat files in the folder
    dat_files = glob.glob(os.path.join(folder_path, '*.dat'))
    all_data = []

    for file in dat_files:
        data = np.fromfile(file, dtype=np.int8)  # 8-bit signed integers
        # data=np.clip(data,-65,0)
        data = data * -1
        # Define out-of-range values
        # out_of_range_indices = np.where(data < -65)[0]

        # # Iterate through each out-of-range index and correct it
        # for idx in out_of_range_indices:
        #     # Determine neighboring indices to take the average
        #     start_idx = max(0, idx - 2)  # Start from 2 positions before, but not less than 0
        #     end_idx = min(len(data), idx + 3)  # Go up to 2 positions after, but not beyond the array size
        #     neighbor_values = np.delete(data[start_idx:end_idx], 2)  # Remove the out-of-range value from the slice
        #     if len(neighbor_values) > 0:
        #         data[idx] = np.mean(neighbor_values)  # Replace the out-of-range value with the mean

        # Apply the first smoothing method (e.g., moving average or Savitzky-Golay)
        # if smoothing_method=='savgol':
        #     data = smooth_data(data, window_size=5, method='savgol')

        # Apply Kalman filter if required
        # if apply_kalman:
        #     data = kalman_filter_filterpy(data)  # Use filterpy for Kalman filter
        # Convert remaining values to positive by multiplying by -1
        
        # # Handle data length (pad or truncate)
        if len(data) < seq_len * input_dim:
            # Pad with zeros if the data is smaller
            data = np.pad(data, (0, seq_len * input_dim - len(data)), 'constant')
        elif len(data) > seq_len * input_dim:
            # Truncate if the data is larger
            data = data[:seq_len * input_dim]

        # Reshape into (seq_len, input_dim)
        reshaped_data = data.reshape(seq_len, input_dim)
        all_data.append(reshaped_data)

        # print(reshaped_data)

    return np.array(all_data)


def smooth_data(data, window_size=5, method='moving_average'):
    """
    Smooth the data using the specified method.
    
    Args:
    - data: The 1D array of data to smooth.
    - window_size: The number of points for smoothing. For moving average, this defines the averaging window.
    - method: 'moving_average' or 'savgol'. Determines the smoothing technique.
    
    Returns:
    - Smoothed data.
    """
    # Ensure the window size is smaller than the data length
    if len(data) < window_size:
        return data  # If the data is too small, return it unchanged

    if method == 'moving_average':
        # Moving average smoothing
        cumsum = np.cumsum(np.insert(data, 0, 0)) 
        smoothed_data = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        # Padding to maintain the original data length
        if len(smoothed_data) > 0:
            smoothed_data = np.pad(smoothed_data, (window_size//2, window_size - window_size//2 - 1), mode='edge')
    
    elif method == 'savgol':
        # Savitzky-Golay filter (window size should be odd, and polyorder < window_size)
        smoothed_data = savgol_filter(data, window_length=5, polyorder=1)

    return smoothed_data

def kalman_filter_filterpy(data):
    """
    Apply Kalman filter smoothing to the data using filterpy.
    
    Args:
    - data: The 1D array of data to smooth using Kalman filter.
    
    Returns:
    - Smoothed data using Kalman filter.
    """
    # Initialize the Kalman filter
    kf = KalmanFilter(dim_x=1, dim_z=1)  # 1D system

    # Kalman filter parameters
    kf.x = np.array([[0.]])  # Initial state
    kf.F = np.array([[1.]])  # State transition matrix
    kf.H = np.array([[1.]])  # Measurement function
    kf.P = np.array([[50]])  # Large initial covariance
    kf.R = np.array([[0.5]])  # Measurement noise
    kf.Q = np.array([[9e-6]])  # Process noise

    smoothed_data = []

    # Apply the Kalman filter iteratively for each data point
    for z in data:
        kf.predict()
        kf.update(z)
        smoothed_data.append(kf.x[0, 0])  # Append the filtered value

    # print(smoothed_data)
    return np.array(smoothed_data)
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
def add_noise(data, noise_factor=1):
    noise = noise_factor * np.random.normal(size=data.shape)
    noisy_data = data + noise
    return np.clip(noisy_data, -65, 0) 

# Set parameters
input_dim = 128  # Number of features
hidden_dim = 256 # LSTM hidden units
seq_len = 60  # Sequence length
# batch_size = 1
learning_rate = 0.0001
epochs = 10500
num_layers = 1# Add more layers
extra_dense_units = 64  # Extra complexity via additional fully connected layers

train_data_folder = 'filtered_output'  # Path to the folder containing .dat files
test_data_folder='./LSTM Test Output'
# Check if GPU is available and use it if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Read and preprocess the training and test data from .dat files
train_data = read_dat_files(train_data_folder, seq_len, input_dim,smoothing_method='savgol', window_size=60,apply_kalman=False)
test_data = read_dat_files(test_data_folder, seq_len, input_dim,smoothing_method='savgol', window_size=60,apply_kalman=True)
train_data_noisy=read_dat_files('./LSTM Train',seq_len,input_dim,smoothing_method='savgol', window_size=60,apply_kalman=False)
test_data_noisy=read_dat_files('./LSTM Test Train',seq_len,input_dim,smoothing_method='savgol', window_size=60,apply_kalman=True)  # You might have a separate folder for test data

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data_noisy = scaler.fit_transform(train_data_noisy.reshape(-1, input_dim)).reshape(-1, seq_len, input_dim)
scaled_test_data_noisy = scaler.transform(test_data_noisy.reshape(-1, input_dim)).reshape(-1, seq_len, input_dim)

# Convert data to PyTorch tensor
train_data_noisy = torch.tensor(scaled_train_data_noisy, dtype=torch.float32)
test_data_noisy = torch.tensor(scaled_test_data_noisy, dtype=torch.float32)


scaled_train_data = scaler.fit_transform(train_data.reshape(-1, input_dim)).reshape(-1, seq_len, input_dim)
scaled_test_data =  scaler.transform(test_data.reshape(-1, input_dim)).reshape(-1, seq_len, input_dim)

# copytestData=test_data

# Convert data to PyTorch tensor
train_data= torch.tensor(scaled_train_data, dtype=torch.float32)
test_data = torch.tensor(scaled_test_data, dtype=torch.float32)
# Move data to GPU if available
# train_data = train_data.to(device)
# test_data = test_data.to(device)
train_dataset = TensorDataset(train_data_noisy, train_data)  # noisy input and clean target
test_dataset = TensorDataset(test_data_noisy, test_data)  # for evaluation

batch_size = 64# Set the batch size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Normalize the dataset using MinMaxScaler
# Define model, loss function, and optimizer
model = LSTMAutoencoder(input_dim, hidden_dim, seq_len, num_layers,extra_dense_units)
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs")
#     model = nn.DataParallel(model)
model.load_state_dict(torch.load("model_194_2_accuracy.pth"))
model = model.to(device)

# Define loss and optimizer
criterion = nn.MSELoss()  # Reconstruction task with mean squared error
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
# Set model save path
#model_save_path = "lstm_autoencoder_model_5.pth"
onnx_model_path = "lstm_autoencoder_model_10.onnx"

# Function to train the model
def train_model(model,train_loader, optimizer, criterion, num_epochs, model_path):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (noisy_batch, clean_batch) in enumerate(train_loader):
            # Move batch to GPU if available
            noisy_batch = noisy_batch.to(device)
            clean_batch=clean_batch.to(device)

            optimizer.zero_grad()
            # Forward pass
            outputs = model(noisy_batch)
            loss = criterion(outputs, clean_batch)
            
            # Backward pass and optimization
           
            loss.backward()

            optimizer.step()
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
        if loss.item()<=0.0167:
            torch.save(model.state_dict(), "model_19_"+str(epoch)+".pth")
            # print(f"Model saved to {model_save_path}")
    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Train model and save it
#model.load_state_dict(torch.load("model_194_2_accuracy.pth"))
#train_model(model,train_loader, optimizer, criterion, epochs, model_save_path)
#model.load_state_dict(torch.load(model_save_path))
if isinstance(model, nn.DataParallel):
    model_to_export = model.module  # Unwrap the model from DataParallel
else:
    model_to_export = model

# # Export the model to ONNX
dummy_input = torch.randn(1, seq_len, input_dim).to(device)  # Dummy input for exporting
torch.onnx.export(model_to_export, dummy_input, onnx_model_path, 
                  input_names=['input'], output_names=['output'], 
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print(f"Model exported to ONNX at {onnx_model_path}")

##Load the ONNX model with ONNX Runtime
#ort_session = ort.InferenceSession(onnx_model_path)

# # Convert test data to NumPy for ONNX inference
# test_data_numpy = scaled_test_data.astype(np.float32)

# Measure inference time for ONNX model
# start_time = time.time()

# Run inference on the test data
# onnx_outputs = ort_session.run(None, {'input': test_data_numpy})



# Display elapsed time for ONNX inference
# model.load_state_dict(torch.load(".pth"))
# Calculate the test loss in PyTorch
model.eval()

with torch.no_grad():
    start_time = time.time()
    # print(test_data_noisy)
    predictions = model(test_data_noisy.to(device))
    print(predictions)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # recon_loss = reconstruction_loss(test_data, predictions)
    # print(f"Reconstruction Loss (MSE): {recon_loss.item():.4f}")
   
    print(f"Time taken for ONNX inference: {elapsed_time:.4f} seconds")

    # Convert predictions back to the original scale
    predictions_numpy = predictions.cpu().detach().numpy()  # Convert to NumPy for inverse transform
    predictions_scaled_back = scaler.inverse_transform(predictions_numpy.reshape(-1, input_dim)).reshape(-1, seq_len, input_dim)

    # Also scale the original test data back for comparison
    test_data_numpy = test_data.cpu().detach().numpy()  # Convert test data to NumPy
    test_data_scaled_back = scaler.inverse_transform(test_data_numpy.reshape(-1, input_dim)).reshape(-1, seq_len, input_dim)
    
    # Calculate reconstruction loss between the original scaled-back data
    print(test_data_scaled_back)
    print(np.round(predictions_scaled_back))
    # print(test_data)
    recon_loss = reconstruction_loss(torch.tensor(test_data_scaled_back), torch.tensor(np.round(predictions_scaled_back)))
    print(f"Reconstruction Loss (MSE) after scaling back: {recon_loss.item():.4f}")
 # Multiply the data by -1 to make it negative
    # test_data_negated =  read_dat_files(train_data_folder, seq_len, input_dim,smoothing_method='savgol', window_size=60,apply_kalman=False)
    # test_data_negated.reshape(-1,60,128)
    test_data_negated=predictions_scaled_back *-1
    test_data_negated=test_data_negated.astype(np.int8)
    # test_data_negated.reshape(-1,60,128)
    
    # Set the initial timestamp (11:20 AM, September 20th, 2024)
    start_time = datetime(2024, 12, 21, 22, 59, 0)

    # Number of samples in your batch
    num_samples = test_data_negated.shape[0]

    for idx in range(num_samples):
        # Get the data matrix for this sample
        data_matrix = test_data_negated[idx]

        # Ensure the data matrix has enough elements to reshape to (60, 128)
        total_elements = data_matrix.size
        required_elements = 60 * 128
        # if total_elements != required_elements:
        #     # You might need to reshape or pad/truncate your data accordingly
        # #    data_matrix = data_matrix.flatten()[:required_elements]  # Truncate if necessary
        data_matrix = data_matrix.reshape(required_elements)
        # else:
        #     # Reshape data_matrix to (60, 128)
        #     data_matrix = data_matrix.reshape(required_elements )

        print(data_matrix)

        # Calculate the timestamp for this file
        timestamp = start_time + timedelta(seconds=idx)
        timestamp_str = timestamp.strftime('%Y%m%d%H%M%S')

        # Construct the filename with .dat extension
        filename = f'01_10_{timestamp_str}.dat'

        # Save the data matrix to a binary .dat file
        data_matrix.tofile(filename)

        print(f"Saved data matrix {idx} to file {filename}")