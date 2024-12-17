import os
import glob
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import remez, lfilter
from binaryfilereader import BinaryFileReader
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def func_conv(u, v):
    """
    Vectorized version of func_conv.
    """
    m = len(u)
    n = len(v)
    conv_uv = np.convolve(u, v)  # length = m+n-1
    # We only need the first n points.
    # Original code divides by min(k+1, m) for each k.
    idxs = np.arange(n)
    denominator = np.where(idxs < m, idxs + 1, m)
    Cum_conv = conv_uv[:n] / denominator
    return Cum_conv


def func_cum4uni_vertical(x):
    """
    Vectorized version of func_cum4uni_vertical.
    Uses cross-correlation and autocorrelation via np.correlate.
    """
    x = x - np.mean(x)  # mean/average of array elements along given axis.
    N = len(x)
    # Autocorrelation of x
    Rxx_full = np.correlate(x, x, mode='full')
    # Cross-correlation of x^3 with x
    C4xx_full = np.correlate(x ** 3, x, mode='full')

    # For m in [0, N-1], we want Rxx(m) = Rxx_full[N-1+m] / (N-m)
    # and C4xx(m) = C4xx_full[N-1+m] / (N-m)
    m_idx = np.arange(N) + (N - 1)
    denom = N - np.arange(N)  # (N-m)

    Rxx = Rxx_full[m_idx] / denom
    C4xx = C4xx_full[m_idx] / denom

    C4x_uv = C4xx - 3 * Rxx * Rxx[0]
    return C4x_uv


def process_file(file_path, output_dir, start_time, idx, plot_first=False):
    """
    Process a single binary file and save the processed data with a timestamped filename.

    Parameters:
    - file_path: Path to the input binary file.
    - output_dir: Directory where the processed file will be saved.
    - start_time: Initial datetime object for timestamping.
    - idx: Index of the current file for timestamp increment.
    - plot_first: Boolean flag to plot the first processed file.
    """
    # Read the binary file
    bin_file_reader = BinaryFileReader()
    bin_file_reader.readFil(file_path)
    raw = bin_file_reader.getArray()
    raw_len = len(raw)

    # Preprocessing
    raw_x_axis_array = np.arange(0, raw_len) * 0.1302
    print(f'*** raw bin size = {len(raw)}')
    nfft = 2 ** int(np.ceil(np.log2(raw_len)))
    xt_fft = fft(raw, nfft) / raw_len
    C4x_uv = func_cum4uni_vertical(raw)
    print(f'C4x_uv length = {len(C4x_uv)}')
    Cum_conv = func_conv(C4x_uv, raw)
    Cum_conv = -1 * Cum_conv
    # FFT parameters
    conv_fft = fft(Cum_conv, nfft) / raw_len
    # ajuste = ((16 / 3) * np.abs(conv_fft)) ** (1 / 32) # (1 / 5)  # amplitude adjustment
    # ajuste_mean = np.mean(ajuste) # Returns a single mean value of the array.
    # ajuste = ajuste - ajuste_mean # Centers the array
    # ajuste[ajuste < 0] = 1 # replace all negative values in the array with 1

    nuevo_num = conv_fft * np.exp(1j * np.angle(conv_fft))
    # Processing
    C4x_uv = func_cum4uni_vertical(raw)
    Cum_conv = func_conv(C4x_uv, raw)
    Cum_conv = -1 * Cum_conv

   # conv_fft = fft(Cum_conv, nfft) / raw_len

    conv_fft = fft(Cum_conv, nfft) / raw_len
    ajuste = ((16 / 3) * np.abs(conv_fft)) ** (1 / 32)  # (1 / 5)  # amplitude adjustment
    # ajuste_mean = np.mean(ajuste) # Returns a single mean value of the array.
    # ajuste = ajuste - ajuste_mean # Centers the array
    ajuste[ajuste < 0] = 1  # replace all negative values in the array with 1

    nuevo_num = conv_fft * np.exp(1j * np.angle(conv_fft))
    # Inverse FFT to get the time-domain signal
    ajuste_ifft = ifft(nuevo_num, nfft) * raw_len
    atn = np.real(ajuste_ifft)
    atn = atn[:raw_len]

    # Envelope detection
    mod_atn = 2 * atn * atn

    # Design the filter
    b = remez(55, [0, 0.1, 0.3, 1], [1, 0], fs=2)

    # Apply the filter
    sal = lfilter(b, 1, mod_atn)
    sal = np.sqrt(np.abs(sal))
    sal[sal == 0] = np.finfo(float).eps  # Avoid division by zero

    # Avoiding the first 100 samples
    # atn[:100] = 0
    atn_final = atn / (sal / 10)  # (sal/100)
    atn_final = atn_final[:raw_len]
    atn_final[:100] = 0
    atn_final[atn_final > 30] = 0

    peaks, _ = find_peaks(atn_final)  # Returns indices of peaks
    peak_values = atn_final[peaks]
    average_peak = np.mean(peak_values)
    peak_max = np.max(atn_final[peaks])

    atn_final_mean = np.mean(atn_final)
    atn_final = atn_final - peak_max
    # atn_final=-atn_final
    window_size = 4  # Larger window -> smoother signal, but more smoothing delay.
    window = np.ones(window_size) / window_size
    #atn_final = np.convolve(atn_final, window, mode='same')
   # elapsed_time = time.time() - start_time
    # Ensure all values are negative
    scaling_factor = -65
   # atn_final = atn_final * scaling_factor  # Scaling factor is negative

    # Reshape to original lengt

    # Reshape to (60, 128) if necessary
    required_elements = 60 * 128  # 7680
    if atn_final.size < required_elements:
        # Pad with zeros if not enough elements
        atn_final = np.pad(atn_final, (0, required_elements - atn_final.size), 'constant')
    elif atn_final.size > required_elements:
        # Truncate if too many elements
        atn_final = atn_final[:required_elements]
    # Reshape
    data_matrix = atn_final.reshape((60, 128))

    # Calculate the timestamp for this file
    timestamp = start_time + timedelta(seconds=idx)
    timestamp_str = timestamp.strftime('%Y%m%d%H%M%S')

    # Construct the filename with .dat extension
    filename = f'01_10_{timestamp_str}.dat'
    output_path = os.path.join(output_dir, filename)

    # Save the data matrix to a binary .dat file
    data_matrix.tofile(output_path)

    logging.info(f"Saved processed file to: {output_path}")

    # Plotting for the first file
    if plot_first and idx == 0:
        # Plot the raw signal and the processed signal
        plt.figure(figsize=(12, 6))

        # Plot raw signal
        plt.subplot(2, 1, 1)
        raw_x_axis_array = np.arange(0, raw_len) * 0.1302
        plt.plot(raw_x_axis_array, raw, 'k')
        plt.title('Raw Signal')
        plt.xlabel('Time(s)')
        plt.ylabel('Amplitude(u)')
        plt.grid(True)

        # Plot processed signal
        plt.subplot(2, 1, 2)
        # Since data_matrix is (60,128), flatten it for plotting
        processed_x_axis_flat = data_matrix.flatten()
        processed_x_axis_time = np.arange(len(processed_x_axis_flat)) * 0.1302  # Adjust as needed
        plt.plot(processed_x_axis_time, processed_x_axis_flat, 'r')
        plt.title('Processed Signal (First File)')
        plt.xlabel('Time(s)')
        plt.ylabel('Amplitude(u)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def main():
    # Define input and output directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, 'input_files')  # Replace 'input_files' with your input folder name
    output_dir = os.path.join(script_dir, 'output_files')  # Replace 'output_files' with your desired output folder name

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    # Define the pattern for binary files (assuming .dat extension)
    file_pattern = '*.dat'
    file_paths = glob.glob(os.path.join(input_dir, file_pattern))

    if not file_paths:
        logging.warning(f"No files found in {input_dir} with pattern {file_pattern}.")
        return

    # Set the initial timestamp (adjust the date and time as needed)
    start_time = datetime(2024, 12, 17, 23, 59, 0)

    # Number of samples in your batch
    num_samples = len(file_paths)

    logging.info(f"Starting processing of {num_samples} files.")

    # Process each file
    for idx, file_path in enumerate(sorted(file_paths)):
        # For plotting the first file, set plot_first=True
        plot_first = (idx == 0)
        process_file(file_path, output_dir, start_time, idx, plot_first=plot_first)

    logging.info("Processing complete.")


if __name__ == "__main__":
    main()
