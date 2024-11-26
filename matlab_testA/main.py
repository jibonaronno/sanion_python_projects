

import numpy as np

def func_conv(kernel, input_array):
    """
    Custom convolution function similar to MATLAB code.
    """
    kernel_len = len(kernel)
    input_len = len(input_array)
    cum_conv = np.zeros(input_len)

    kernel_indexes = []
    sum_arrays = []
    #F = 0
    for k in range(input_len):
        F = 0
        kernel_loop_count = 0
        kernel_indexes.clear()
        sum_arrays.clear()
        for j in range(max(0, k + 1 - input_len), min(k + 1, kernel_len)):
            F += kernel[j] * input_array[k - j]
            kernel_loop_count += 1
            sum_arrays.append((kernel[j], input_array[k - j]))
            kernel_indexes.append((max(0, k + 1 - input_len), min(k + 1, kernel_len), (k + 1 - input_len), ('j=',j)))
        cum_conv[k] = F / min(k + 1, kernel_len)
        # print(f'kernel_loop_count = {kernel_loop_count} - k={k} {kernel_indexes} F={F} cum_conv[k]={cum_conv[k]}')
        print(f'kernel_loop_count = {kernel_loop_count} - k={k} sum_arrays=SUM({sum_arrays})  F={F} cum_conv[k]={cum_conv[k]}')
        # print((min(k+1, m)))
    #print(F)
    return cum_conv

def convolve(input_array, kernel):
    kernel = kernel[::-1]  # Reverse kernel for convolution
    output = []
    for i in range(len(input_array) - len(kernel) + 1):
        result = sum(input_array[i + j] * kernel[j] for j in range(len(kernel)))
        output.append(result)
    return output

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_array = [1, 2, 3, 4, 5, 6, 7, 8, 8, 1, 4, 9]
    kernel = [1, 0, -1]
    #print(convolve(input_array, kernel))
    result = func_conv(kernel, input_array)
    print(f'{len(result)} = \n')
    print(result)

