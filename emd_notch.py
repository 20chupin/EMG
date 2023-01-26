import emd

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from pyomeca import Analogs

# Signal
fs = 10000.0  # Sample frequency (Hz) ("Sensor 13.IM EMG13" was sampled at
# 2000 Hz but since the time step is 1e-5, the value is the same during 5 
# time iterations)

f0 = 33.33333 # FES frequency

data_path = "../EMG_Biceps/EMG_with_FES_and_Muscle_Force/EMG_with_FES_and_Muscle_Force.c3d"
channels = ["Sensor 13.IM EMG13"]

# Stage 0: Band pass filtering [20, 150 Hz]
data = Analogs.from_c3d(data_path, usecols=channels).meca.band_pass(order=2, cutoff=[20, 150])

time = data.time[51000:56000]
sEMG = np.asarray(data.sel(channel="Sensor 13.IM EMG13")[51000:56000])

sEMG_100 = 100 * sEMG # I don't know why yet but it doesn't work without
# multiplying by 100 (my guess is that the stop condition is too high for our
# data)

# Stage I: Empirical Mode Decomposition (EMD) -> I need to work on the stop condition (see article SD > 0.1)
imf = emd.sift.sift(sEMG_100)

# Stage II: Identification of Artifact IMFs
Sd = np.std(imf, axis=0)
Md = np.median(Sd)
Th_sd = 3 * Md

names = [str(i) for i in range(1, len(Sd)+1)]
plt.bar(names, Sd)
plt.plot(names, [Th_sd]*len(names), 'k--', label="Threshold")
plt.xlabel("IMF No.")
plt.ylabel("SD")
plt.legend()
plt.show()

IMF_art = imf.T[Sd > Th_sd]
IMF_emg = imf.T[Sd < Th_sd]

# Stage III: Application of Notch Filtering and final output
S_emg = np.sum(IMF_emg, axis=0)

Q = 10

y = []
SNR=[]

for IMF_arti in IMF_art:
    S_art_int = IMF_arti
    for i in range(1, 13):
        b, a = signal.iirnotch(i * f0, i * Q, fs)
        S_art_int = signal.lfilter(b, a, S_art_int)
    y.append(S_emg + S_art_int)
    SNR.append(np.sum(sEMG_100**2/(sEMG_100-y[-1])**2))

SNR = np.asarray(SNR)
imax = SNR.argmax()
cleaned_EMG = y[imax]

# Applying only the Notch filter
S_notch = sEMG_100
for i in range(1, 13):
    b, a = signal.iirnotch(i * f0, Q, fs)
    S_notch = signal.lfilter(b, a, S_notch)

# Applying only the comb filter
b, a = signal.iircomb(f0, Q, ftype='notch', fs=fs)
S_comb = signal.lfilter(b, a, sEMG_100)

plt.plot(time, sEMG_100/100, label="Raw data")
plt.plot(time, cleaned_EMG/100, label=f"EMD-Notch, Q={Q}")
plt.plot(time, S_emg/100, label="EMD + Th_sd")
plt.plot(time, S_notch/100, label=f"Notch, Q={Q}")
plt.plot(time, S_comb/100, label=f"Comb filter, Q={Q}")
plt.legend()
plt.show()