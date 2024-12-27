
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import zoom
import threading


def save_image(matrix, file_name):
    plt.imsave(file_name, matrix[::-1],cmap="hot")
    # print(f"Image saved successfully as {file_name}.")

def drawTrainingImage(data2,counter):
    def buildUpPicture(oneDimArr, matrix):
        for idx in range(len(oneDimArr)):
            idxForArr = oneDimArr[idx]
            matrix[idxForArr - 1][idx] = 1
        return matrix

    subArrayLen = 128
    maxheight = 65
    newMatrix = np.zeros((65, 128))
    data = data2+65
    # print(data.shape)
    data=np.where(data<1,0,data)
    # print(data.shape)
    # Move the tensor to the GPU
    # Reshape the tensor
    a = data.reshape(-1, 128)
    mask = a <= maxheight
    # newMatrix = np.zeros_like(a)
    for j in range(a.shape[1]):
        column_data = a[:, j]  # Get the j-th column of a
        counts = np.bincount(column_data[mask[:, j]] - 1, minlength=maxheight)
        newMatrix[:, j] = counts
    norm = colors.LogNorm()
    cmap = plt.get_cmap("hot")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba_matrix = sm.to_rgba(newMatrix, bytes=True)

    # Convert RGBA to RGB and set the background to white
    rgb_matrix = rgba_matrix[:, :, :3]
    alpha_channel = rgba_matrix[:, :, 3] / 255.0

    # Create a white background
    white_background = np.ones_like(rgb_matrix) * 255

    # Blend the RGB matrix with the white background
    final_matrix = (rgb_matrix * alpha_channel[:, :, None] + white_background * (1 - alpha_channel[:, :, None])).astype(np.uint8)
    zoom_factor = (1,4.20,4.5)

# Zoom with nearest neighbor interpolation (order=1)
    zoomed_matrix = zoom(np.transpose(final_matrix,(2,0,1)), zoom_factor, order=1)
    # print(zoomed_matrix.shape)
    image_saving_thread = threading.Thread(target=save_image, args=(np.transpose(zoomed_matrix,(1,2,0)), "R:\\file_"+ str(counter)+".jpg"))
    image_saving_thread.start()

    return newMatrix
def pad_or_truncate_data(data: bytes, length: int) -> bytes:
    # If data is shorter than the desired length, pad with zeros
    if len(data) < length:
        data = data.ljust(length, b'\x00')
    # If data is longer than the desired length, truncate it
    elif len(data) > length:
        data = data[:length]
    return data
async def DrawImage(binaryData,counter):
    processed_data = pad_or_truncate_data(binaryData, 460800)

    dataArr = np.frombuffer(processed_data,dtype=np.int8)
    newMatrix=drawTrainingImage(dataArr,counter)
    return newMatrix

