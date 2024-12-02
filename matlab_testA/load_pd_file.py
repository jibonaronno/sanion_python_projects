import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import remez, lfilter
import time

def func_conv(u, v):
    m = len(u)
    n = len(v)
    Cum_conv = np.zeros(n)
    for k in range(n):
        F = 0
        for j in range(max(0, k - n + 1), min(k + 1, m)):
            F += u[j] * v[k - j]
        Cum_conv[k] = F / min(k + 1, m)
    return Cum_conv

def func_cum4uni_vertical(x):
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
Fs = 1000  # Sample frequency
fc = 50    # Carrier frequency
T = 1 / Fs # Sample period
KB = 1024
L = 1 * 1024
t = np.arange(0, L) * T
A = 1
nfft = 2 ** int(np.ceil(np.log2(L)))
wc = 2 * np.pi * fc

xt = np.cos(2 * np.pi * fc * t + np.pi / 4)

plt.figure()
plt.plot(t, xt, 'k')
plt.title('Useful Signal')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude(u)')
plt.grid(True)
plt.show()

xt_fft = fft(xt, nfft) / L
f = Fs * np.linspace(0, 1, nfft)

plt.figure()
plt.plot(f, np.abs(xt_fft), 'k')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude(u)')
plt.title('Useful Signal Spectrum')
plt.grid(True)
plt.show()

np.random.seed(0)  # set seed for reproducibility
noise = 2.2 * np.random.randn(len(xt))
xt_noise = xt + noise

plt.figure()
plt.plot(t, xt_noise, 'k')
plt.title('Noisy Signal')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude(u)')
plt.grid(True)
plt.show()

xt_noise_fft = fft(xt_noise, nfft) / L

plt.figure()
plt.plot(f, np.abs(xt_noise_fft), 'k')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude(u)')
plt.title('Noisy Signal Spectrum')
plt.grid(True)
plt.show()

start_time = time.time()
C4x_uv = func_cum4uni_vertical(xt_noise)
Cum_conv = func_conv(C4x_uv, xt_noise)
Cum_conv = -1 * Cum_conv

conv_fft = fft(Cum_conv, nfft) / L

ajuste = ((16 / 3) * np.abs(conv_fft)) ** (1 / 5)  # amplitude adjustment

ajuste_mean = np.mean(ajuste)
ajuste = ajuste - ajuste_mean
ajuste[ajuste < 0] = 1

nuevo_num = ajuste * np.exp(1j * np.angle(conv_fft))

ajuste_ifft = ifft(nuevo_num, nfft) * L
atn = np.real(ajuste_ifft)
atn = atn[:L]

mod_atn = 2 * atn * atn  # envelope detection

# Design the filter
b = remez(21, [0, 0.03, 0.1, 1], [1, 0], fs=2)

# Apply the filter
sal = lfilter(b, 1, mod_atn)
sal = np.sqrt(np.abs(sal))
sal[sal == 0] = np.finfo(float).eps  # Avoid division by zero
atn_final = atn / sal
atn_final = atn_final[:L]

elapsed_time = time.time() - start_time

plt.figure()
plt.plot(t, atn_final, 'r')
plt.title('Comparison in time Useful Signal - Proposed algorithm')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude(u)')
plt.grid(True)
plt.show()

atn_final_fft = fft(atn_final, nfft) / L

plt.figure()
plt.plot(f, np.abs(xt_fft), 'b', label='Useful')
plt.plot(f, np.abs(atn_final_fft), 'r', label='Output')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude(u)')
plt.title('Comparison of the spectrum of the Useful Signal - Proposed algorithm')
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(f, np.abs(xt_noise_fft), 'b', label='Noisy')
plt.plot(f, np.abs(atn_final_fft), 'r', label='Output')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude(u)')
plt.title('Comparison of the spectrum of the Noisy Signal - Proposed algorithm')
plt.grid(True)
plt.legend()
plt.show()

# Compute power
px = np.mean(np.abs(xt) ** 2)
pn = np.mean(np.abs(noise) ** 2)
SNR = 10 * np.log10(px / pn)

pxot = np.mean(np.abs(atn_final) ** 2)
pno = pxot - px
SNRO = 10 * np.log10(px / pno)

print(f"Length of xt: {len(xt)}")
print(f"Length of xt_noise: {len(xt_noise)}")

CORRE_INICIAL = np.corrcoef(xt, xt_noise)
CORRE_FINAL = np.corrcoef(xt, atn_final)

print("Initial Correlation Coefficient between xt and xt_noise:")
print(CORRE_INICIAL)
print("Final Correlation Coefficient between xt and atn_final:")
print(CORRE_FINAL)
