import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import firwin, lfilter

def func_conv(u, v):
    """
    Custom convolution function similar to MATLAB code.
    """
    m = len(u)
    n = len(v)
    cum_conv = np.zeros(n)

    for k in range(n):
        F = 0
        for j in range(max(0, k + 1 - n), min(k + 1, m)):
            F += u[j] * v[k - j]
        cum_conv[k] = F / min(k + 1, m)
    return cum_conv

def func_cum4uni_vertical(x):
    """
    Calculates the fourth-order cumulant for a signal.
    """
    x = x - np.mean(x)
    N = len(x)
    C4xx = np.zeros(N)
    Rxx = np.zeros(N)

    for m in range(N):
        F = 0
        F1 = 0
        for k in range(N - m):
            F += x[k] * x[k] * x[k] * x[k + m]
            F1 += x[k] * x[k + m]
        C4xx[m] = F / (N - m)
        Rxx[m] = F1 / (N - m)
    C4x_uv = C4xx - 3 * Rxx * Rxx[0]
    return C4x_uv

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

# Add noise to signal
noise = 2.2 * np.random.randn(L)
xt_noise = xt + noise

plt.figure()
plt.plot(t, xt_noise, 'k')
plt.title('Noisy Signal')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude(u)')
plt.grid()
plt.show()

# FFT of noisy signal
xt_noise_fft = fft(xt_noise, nfft) / L

plt.figure()
plt.plot(f, np.abs(xt_noise_fft), 'k')
plt.title('Noisy Signal Spectrum')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude(u)')
plt.grid()
plt.show()

# Process noisy signal
C4x_uv = func_cum4uni_vertical(xt_noise)
Cum_conv = func_conv(C4x_uv, xt_noise)
Cum_conv = -1 * Cum_conv

conv_fft = fft(Cum_conv, nfft) / L
ajuste = ((16 / 3) * np.abs(conv_fft)) ** (1 / 5)
ajuste_mean = np.mean(ajuste)

# Correction by mean
ajuste = np.maximum(ajuste - ajuste_mean, 1)

nuevo_num = ajuste * np.exp(1j * np.angle(conv_fft))
ajuste_ifft = ifft(nuevo_num, nfft) * L
atn = np.real(ajuste_ifft)

mod_atn = 2 * atn**2  # Envelope detection
b = firwin(21, [0.03, 0.1], pass_zero=False, fs=2)
sal = np.sqrt(lfilter(b, 1, mod_atn))
atn_final = atn / sal

plt.figure()
plt.plot(t, atn_final, 'r')
plt.title('Comparison in time: Useful Signal vs. Proposed Algorithm')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude(u)')
plt.grid()
plt.show()

# FFT of final signal
atn_final_fft = fft(atn_final, nfft) / L

plt.figure()
plt.plot(f, np.abs(xt_fft), 'b', label='Useful')
plt.plot(f, np.abs(atn_final_fft), 'r', label='Output')
plt.title('Comparison of Spectrums: Useful Signal vs. Proposed Algorithm')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude(u)')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(f, np.abs(xt_noise_fft), 'b', label='Noisy')
plt.plot(f, np.abs(atn_final_fft), 'r', label='Output')
plt.title('Comparison of Spectrums: Noisy Signal vs. Proposed Algorithm')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude(u)')
plt.legend()
plt.grid()
plt.show()

# SNR calculations
px = np.mean(np.abs(xt)**2)
pn = np.mean(np.abs(noise)**2)
SNR = 10 * np.log10(px / pn)

pxot = np.mean(np.abs(atn_final)**2)
pno = pxot - px
SNRO = 10 * np.log10(px / pno)

CORRE_INICIAL = np.corrcoef(xt, xt_noise)[0, 1]
CORRE_FINAL = np.corrcoef(xt, atn_final)[0, 1]

print(f"Initial Correlation: {CORRE_INICIAL}")
print(f"Final Correlation: {CORRE_FINAL}")
print(f"SNR: {SNR:.2f} dB")
print(f"Output SNR: {SNRO:.2f} dB")
