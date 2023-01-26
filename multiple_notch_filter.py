import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from pyomeca import Analogs


def multiple_notch(sig, f0, fs, Q, n=12):
    notch_filtered = sig

    for i in range(1, n + 1):
        b, a = signal.iirnotch(i * f0, i * Q, fs)
        notch_filtered = signal.lfilter(b, a, notch_filtered)

    return notch_filtered

# Signal
fs = 2000.0

data_path_FES = "../EMG_Biceps/EMG_with_FES/EMG_Biceps_with_FES.c3d"
channels = ["Sensor 13.IM EMG13"]

emg = Analogs.from_c3d(data_path_FES, usecols=channels)

# I'm working with numpy for the moment
time = np.asarray(emg.time[50500:52400])
sEMG = np.asarray(emg.sel(channel='Sensor 13.IM EMG13')[50500:52400])

# Filters
f0 = 33.333333
Q = 10.0

# Comb
b, a = signal.iircomb(f0, 10, ftype='notch', fs=fs)
EMG_comb = signal.lfilter(b, a, sEMG)

# Multiple Notch
EMG_mnotch = multiple_notch(sEMG, f0, fs, Q)

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
axs[0].plot(time, sEMG)
axs[0].set_title("Raw data")
axs[1].plot(time, EMG_comb)
axs[1].set_title("Comb filter")
axs[2].plot(time, EMG_mnotch)
axs[2].set_title("Multiple Notch")
axs[2].set_xlabel("Time (s)")
axs[0].set_ylabel("EMG (V)")
axs[1].set_ylabel("EMG (V)")
axs[2].set_ylabel("EMG (V)")
plt.show()