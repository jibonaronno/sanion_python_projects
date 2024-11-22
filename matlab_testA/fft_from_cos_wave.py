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

# FFT of useful signal
nfft = 2**np.ceil(np.log2(L)).astype(int)
xt_fft = fft(xt, nfft) / L
f = Fs * np.linspace(0, 1, nfft)

plt.figure()
plt.plot(f, np.abs(xt_fft), 'k')
plt.title('Useful Signal Spectrum')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude(u)')
plt.grid()
plt.show()
