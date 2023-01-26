import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from pyomeca import Analogs


# Signal
fs = 10000.0  # Sample frequency (Hz) ("Sensor 13.IM EMG13" was sampled at
# 2000 Hz but since the time step is 1e-5, the value is the same during 5 
# time iterations)

data_path_FES_and_MF = "../EMG_Biceps/EMG_with_FES_and_Muscle_Force/EMG_with_FES_and_Muscle_Force.c3d"
data_path_FES = "../EMG_Biceps/EMG_with_FES/EMG_Biceps_with_FES.c3d"
channels = ["Sensor 13.IM EMG13", "Electric Current.FES"]

emg = Analogs.from_c3d(data_path_FES_and_MF, usecols=channels).meca.band_pass(order=2, cutoff=[20, 300])
emg_without_MF = Analogs.from_c3d(data_path_FES, usecols=channels).meca.band_pass(order=2, cutoff=[20, 300])

# I'm working with numpy for the moment
time = np.asarray(emg.time[51000:56000])
sEMG = np.asarray(emg.sel(channel='Sensor 13.IM EMG13')[51000:56000])
time_without_MF = np.asarray(emg_without_MF.time[51000:52000])
sEMG_without_MF = np.asarray(emg_without_MF.sel(channel='Sensor 13.IM EMG13')[51000:52000])

# FFT of raw data
fourier = np.fft.fft(sEMG)
n = sEMG.size
freq = np.fft.fftfreq(n, d=1/fs)

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
fig.suptitle("FFT")
axs[0].set_title("Raw data")
axs[0].plot(freq, fourier.real, label="real")
axs[0].plot(freq, fourier.imag, label="imag")
axs[0].legend()

# Filter
f0 = 33.333333  # Frequency to be removed from signal (Hz) (FES frequency,
# fs must be a multiple of f0 for this implementation to work)
Q = 10.0  # Quality factor

b, a = signal.iircomb(f0, 10, ftype='notch', fs=fs) # ftype=notch car on veut
# couper des fréquences et pas garder des `peaks`

EMG = signal.lfilter(b, a, sEMG)

# FFT of filtered data
fourier = np.fft.fft(EMG)
n = EMG.size
freq = np.fft.fftfreq(n, d=1/fs)

axs[1].set_title("Filtered data")
axs[1].plot(freq, fourier.real, label="real")
axs[1].plot(freq, fourier.imag, label="imag")
axs[1].legend()

plt.figure(2)
plt.plot(time, sEMG, label="Raw data")
plt.plot(time, EMG, label="Filtered data")
plt.xlabel("Time (s)")
plt.ylabel("EMG (V)")
plt.title(f"EMG before and after filtering, Q ={Q}")
plt.legend()

plt.show()

# Comparison of the two files
fig, axs = plt.subplots(2, 1, sharex=False, sharey=True)
fig.suptitle("Comparison of the EMGs with and without muscle force")
axs[0].plot(time, sEMG, label="Raw data")
axs[0].plot(time, EMG, label="Filtered data")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("EMG (V)")
axs[0].set_title(f"EMG before and after filtering, Q ={Q}, with muscle force")
axs[0].legend()

b_without_MF, a_without_MF = signal.iircomb(f0, 10, ftype='notch', fs=2000)
EMG_without_MF = signal.lfilter(b_without_MF, a_without_MF, sEMG_without_MF)
axs[1].plot(time_without_MF, sEMG_without_MF, label="Raw data")
axs[1].plot(time_without_MF, EMG_without_MF, label="Filtered data")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("EMG (V)")
axs[1].set_title(f"EMG before and after filtering, Q ={Q}, without muscle force")
axs[1].legend()

plt.show()

# Comparison between different Q factors

plt.plot(time, sEMG, label="Raw data")

for Q in [2, 5, 10, 30]:
    b, a = signal.iircomb(f0, Q, ftype='notch', fs=fs)
    EMG = signal.lfilter(b, a, sEMG)
    plt.plot(time, EMG, label=f"Q = {Q}")

plt.xlabel("Time (s)")
plt.ylabel("EMG (V)")
plt.title("EMG before and after filtering for different quality factors")
plt.legend()

plt.show()

# Comparison between the application of the filter with and without the band pass
emg_unfiltered = Analogs.from_c3d(data_path_FES_and_MF, usecols=channels)

sEMG_unfiltered = np.asarray(emg_unfiltered.sel(channel='Sensor 13.IM EMG13')[51000:56000])

Q = 10.0  # Quality factor

b, a = signal.iircomb(f0, Q, ftype='notch', fs=fs)

EMG = signal.lfilter(b, a, sEMG)
EMG_unfiltered = signal.lfilter(b, a, sEMG_unfiltered)

plt.plot(time, sEMG, label="Raw data band-pass filtered")
plt.plot(time, sEMG_unfiltered, label="Raw data unfiltered")
plt.plot(time, EMG, label="Filtered data from band_pass filtered")
plt.plot(time, EMG_unfiltered, label="Filtered data from unfiltered")
plt.xlabel("Time (s)")
plt.ylabel("EMG (V)")
plt.title(f"EMG before and after filtering with or without band-pass filtering, Q ={Q}")
plt.legend()

plt.show()