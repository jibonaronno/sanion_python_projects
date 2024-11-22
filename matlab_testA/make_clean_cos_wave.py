import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import firwin, lfilter

# Main script
Fs = 1000  # Sampling frequency
fc = 50    # Carrier frequency
T = 1 / Fs  # Sampling period
L = 1024    # Signal length
t = np.arange(0, L) * T

# Generate useful signal
xt = np.cos(2 * np.pi * fc * t + np.pi / 4)

plt.figure()
plt.plot(t, xt, 'k')
plt.title('Useful Signal')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude(u)')
plt.grid()
plt.show()