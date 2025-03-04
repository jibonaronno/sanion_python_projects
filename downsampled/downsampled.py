import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import zoom
def downsample_with_conv_avg(input_data: np.ndarray) -> np.ndarray:
    """
    Downsample a 1024×60 matrix (int8) to 128×60 using a convolution average filter.
    Returns a (128×60) np.ndarray (float32 before conversion/clamping).
    """
    # Convert to float32 for PyTorch
    # input_data has shape (1024, 60) => interpret as (batch=1, channels=60, length=1024)
    tensor_in = torch.from_numpy(input_data.astype(np.float32)).transpose(0, 1).unsqueeze(0)
    # tensor_in.shape => (1, 60, 1024)

    # Define a standard 1D convolution for downsampling:
    #   - kernel_size=8, stride=8
    #   - groups=60 => each channel is filtered independently
    #   - weights = 1/8 => average over 8 samples
    conv_down = nn.Conv1d(
        in_channels=60,
        out_channels=60,
        kernel_size=8,
        stride=8,
        groups=60,
        bias=False
    )

    # Initialize weights => average filter
    with torch.no_grad():
        conv_down.weight.fill_(1.0 / 8.0)

    # Perform the downsampling convolution (no gradient tracking)
    with torch.no_grad():
        downsampled = conv_down(tensor_in)  # shape => (1, 60, 128)

    # Convert back to NumPy => shape (128, 60)
    downsampled_2d = downsampled.squeeze(0).transpose(0, 1).detach().cpu().numpy()

    return downsampled_2d  # float32

def create_heatmap_image(downsampled_data: np.ndarray, output_image: str):
    """
    Takes a 128×60 float (or int) array, assumed in range [-65..0] (or similar),
    shifts to [0..65], computes a 65×128 bin-count histogram, then saves a
    log-scaled "hot" colormap image as output_image (JPG/PNG).
    """
    # 1) Shift from [-65..0] to [0..65] if that's your data's range:
    #    Adjust if your data has a different range.
    shifted = downsampled_data

    # 2) Reshape or transpose so columns = 128 if we want "column_data" = 60 samples
    #    Typically, if shape is (128, 60), and we want each column to have 60 rows,
    #    let's transpose => shape (60, 128).
    a = shifted.transpose()  # => shape (60, 128)

    # 3) For each of the 128 columns, compute a bin-count over the range [0..65]
    newMatrix = np.zeros((65, 128), dtype=int)
    for j in range(a.shape[1]):  # j in [0..127]
        column_data = a[:, j].astype(int)
        counts = np.bincount(column_data, minlength=65)
        newMatrix[:, j] = counts[:65]

    # 4) Apply log-scaled "hot" colormap
    norm = colors.LogNorm()  
    cmap = plt.get_cmap("hot")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    rgba_matrix = sm.to_rgba(newMatrix, bytes=True)  # shape => (65, 128, 4)

    # 5) Alpha-blend on a white background
    rgb_matrix = rgba_matrix[:, :, :3]
    alpha_channel = rgba_matrix[:, :, 3] / 255.0
    white_bg = np.ones_like(rgb_matrix) * 255
    final_matrix = (
        rgb_matrix * alpha_channel[:, :, None] +
        white_bg * (1 - alpha_channel[:, :, None])
    ).astype(np.uint8)
    zoom_factor = (1,4.20,4.5)
    temp = np.transpose(final_matrix, (2, 0, 1))  # shape (3, H, W)

    #temp = np.transpose(final_matrix, (2, 0, 1))  # shape (3, H, W)
    zoomed_matrix = zoom(temp, zoom_factor, order=1)  # still (3, newH, newW)

    # 2) Reorder back to (newH, newW, 3)
    zoomed_matrix = np.transpose(zoomed_matrix, (1, 2, 0))  # shape => (newH, newW, 3)

    # 3) If you want to flip vertically, do [::-1, :, :]
    zoomed_matrix = zoomed_matrix[::-1, :, :]  # shape => (newH, newW, 3)

    # 4) Save without a colormap (already an RGB image)
    plt.imsave(output_image, zoomed_matrix)
    # 6) Save as an image
    # plt.imsave(output_image, zoomed_matrix[::-1], cmap="hot")
    print(f"Saved heatmap image to {output_image}")


def process_single_file(input_file: str, output_image: str):
    """
    Reads a .dat file of 61440 int8 values (1024×60), downsamples it to 128×60
    using a conv average filter, then generates a heatmap image and saves it.
    """
    # --- 1) Read .dat file (1024×60 = 61440 bytes, int8) ---
    with open(input_file, "rb") as f:
        raw = f.read()
    data_array = np.frombuffer(raw, dtype=np.int8)
    if data_array.size != 61440:
        raise ValueError(f"Expected 61440 int8 values, found {data_array.size}.")

    # Reshape => (1024, 60)
    data_array = data_array.reshape(1024, 60)

    # --- 2) Downsample to (128, 60) ---
    downsampled_data = downsample_with_conv_avg(data_array)  # float32

    # (Optionally clamp to [-65..0] if your data is known to be in that range originally.
    #  or skip if your data is in some other range. Adjust logic in create_heatmap_image.)

    # --- 3) Create & save a heatmap image ---
    create_heatmap_image(downsampled_data, output_image)


def main():
    import sys
    if len(sys.argv) < 3:
        print("Usage: python downsample_and_draw.py <input.dat> <output.jpg>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_image = sys.argv[2]

    process_single_file(input_file, output_image)

if __name__ == "__main__":
    main()
