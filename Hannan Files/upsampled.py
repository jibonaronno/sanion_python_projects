import os
import torch
import torch.nn as nn
import numpy as np

def upsample_with_conv_avg(input_data: np.ndarray) -> np.ndarray:
    """
    Upsample a 128×60 matrix (int8) to 1024×60 using a convolution average filter.
    Returns a 1024×60 np.ndarray (int8).
    
    Parameters:
    -----------
    input_data : np.ndarray
        Shape (128, 60), dtype int8 (or any integer type).
    
    Returns:
    --------
    np.ndarray
        Shape (1024, 60), dtype=int8
    """
    # Convert to float32 in PyTorch for the convolution step
    # Shape is (128, 60) => interpret as (batch=1, channels=60, length=128) in PyTorch
    tensor_in = torch.from_numpy(input_data.astype(np.float32)).transpose(0, 1).unsqueeze(0)
    # tensor_in shape: (1, 60, 128)

    # Define a transposed 1D convolution layer for upsampling by factor of 8
    # - in_channels=60, out_channels=60
    # - kernel_size=8, stride=8
    # - groups=60 => separate filter per channel
    # - bias=False => we only want average filtering
    conv_trans = nn.ConvTranspose1d(
        in_channels=60,
        out_channels=60,
        kernel_size=8,
        stride=8,
        groups=60,
        bias=False
    )

    # Initialize weights for an 8-tap average filter => 1/8 each
    with torch.no_grad():
        conv_trans.weight.fill_(1.0 / 8.0)

    # Apply the transposed convolution
    upsampled = conv_trans(tensor_in)  # shape => (1, 60, 1024)

    # Convert back to (1024, 60) in numpy
    # upsampled is float32, shape (1,60,1024)
    upsampled_2d = upsampled.squeeze(0).transpose(0,1).detach().cpu().numpy()

    # Optionally clamp to int8 range [-128..127], then convert to int8
    upsampled_2d = np.clip(upsampled_2d, -65, 0).astype(np.int8)

    return upsampled_2d


def process_single_file(input_path: str, output_path: str):
    """
    Reads a .dat file of 7680 int8 values (128×60), upscales to 1024×60,
    and writes the result as int8 to another .dat file.
    """
    # Read all bytes
    with open(input_path, "rb") as f:
        raw = f.read()

    # Convert to numpy int8
    data_array = np.frombuffer(raw, dtype=np.int8)
    if data_array.size != 7680:
        raise ValueError(f"File '{input_path}' has {data_array.size} int8s, expected 7680.")

    # Reshape to 128×60
    data_array = data_array.reshape(128, 60)

    # Upsample
    upsampled_data = upsample_with_conv_avg(data_array)  # (1024, 60) int8

    # Flatten to 1D buffer (row-major) before saving
    #  => 1024 * 60 = 61440 int8
    upsampled_flat = upsampled_data.reshape(-1)

    # Save to output file
    with open(output_path, "wb") as f:
        f.write(upsampled_flat.tobytes())

    print(f"Upsampled '{input_path}' => '{output_path}'  (shape: 1024×60)")


def batch_upsample_folder(input_dir: str, output_dir: str):
    """
    Scans input_dir for .dat files, upscales each from 128×60 to 1024×60,
    and saves the output .dat in output_dir with the same base name.
    """
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".dat")]
    files.sort()

    for fname in files:
        in_path = os.path.join(input_dir, fname)
        base, _ = os.path.splitext(fname)
        out_path = os.path.join(output_dir, base + "_upsampled.dat")

        process_single_file(in_path, out_path)

def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python batch_upsample_convolution.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    batch_upsample_folder(input_dir, output_dir)

if __name__ == "__main__":
    main()
