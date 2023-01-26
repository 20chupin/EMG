import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

f0 = 33.333333
fs = 1000

# Comb filter, frequency response
for Q in [2, 5, 10, 30]:
    b, a = signal.iircomb(f0, Q, ftype='notch', fs=fs)
    freq, h = signal.freqz(b, a, fs=fs)
    response = abs(h)
    # To avoid divide by zero when graphing
    response[response == 0] = 1e-20
    # Plot
    plt.plot(freq, 20*np.log10(abs(response)), label=f"Q = {Q}")
plt.title("Frequency Response - Comb filter")
plt.ylabel("Amplitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylim([-50, 10])
plt.legend()
plt.show()

# Notch filter, frequency response
for Q in [2, 5, 10, 30]:
    b, a = signal.iirnotch(f0, Q, fs=fs)
    freq, h = signal.freqz(b, a, fs=fs)
    response = abs(h)
    # To avoid divide by zero when graphing
    response[response == 0] = 1e-20
    # Plot
    plt.plot(freq, 20*np.log10(abs(response)), label=f"Q = {Q}")
plt.title("Frequency Response - Notch filter")
plt.ylabel("Amplitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylim([-50, 10])
plt.legend()
plt.show()


# Multiple Notch filters
time = np.arange(0, 1, 1/fs)
sig = np.cos(2*np.pi * f0 * time) + 5 + np.sin(2*np.pi * 20 * time) + np.sin(2*np.pi * 2 * f0 * time) + 0.5*np.cos(2*np.pi * 14 * f0 * time)
# Comparison with comb filter
Q = 10.0
b, a = signal.iircomb(f0, Q, ftype='notch', fs=fs)
comb_filtered = signal.lfilter(b, a, sig)
freq, h = signal.freqz(b, a, fs=fs)
response = abs(h)
# To avoid divide by zero when graphing
response[response == 0] = 1e-20
# Plot
plt.plot(freq, 20*np.log10(abs(response)), label=f"comb filter - f0 = {f0}")
notch_filtered = sig
for i in range(1, 13):
    b, a = signal.iirnotch(i * f0, i * Q, fs)
    notch_filtered = signal.lfilter(b, a, notch_filtered)
    freq, h = signal.freqz(b, a, fs=fs)
    response = abs(h)
    # To avoid divide by zero when graphing
    response[response == 0] = 1e-20
    # Plot
    plt.plot(freq, 20*np.log10(abs(response)), label=f"f0 = {i * f0}")
plt.title("Frequency Response - multiple Notch filters, Q = f / f0 - comparison with the comb filter, Q = 10")
plt.ylabel("Amplitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylim([-50, 10])
plt.legend()
plt.show()

plt.plot(time, sig, label="raw")
plt.plot(time, comb_filtered, label="comb")
plt.plot(time, notch_filtered, label='multiple notch')
plt.xlabel("Time (s)")
plt.title("Comparision of the comb filter and the multiple notch filter on a known signal")
plt.legend()
plt.show()