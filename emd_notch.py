import emd
import copy

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from pyomeca import Analogs

# Signal
fs = 2000.0

f0 = 33.33333 # FES frequency

data_path =  "../EMG_Biceps/EMG_with_FES_and_Muscle_Force/EMG_with_FES_and_Muscle_Force.c3d"
channels = ["Sensor 13.IM EMG13"]

# Stage 0: Band pass filtering [20, 150 Hz]
data = Analogs.from_c3d(data_path, usecols=channels)#.meca.band_pass(order=2, cutoff=[20, 150])

time = data.time[51000:56000:5]
sEMG = np.asarray(data.sel(channel="Sensor 13.IM EMG13")[51000:56000:5])

b, a = signal.butter(2, [20, 350], btype='bandpass', fs=fs)
sEMG = signal.lfilter(b, a, sEMG)

# Stage I: Empirical Mode Decomposition (EMD) -> I need to work on the stop condition (see article SD > 0.1)
imf = emd.sift.sift(sEMG, sift_thresh=1e-15)

# Stage II: Identification of Artifact IMFs
Sd = np.std(imf, axis=0)
Md = np.median(Sd)
Th_sd = 3 * Md

# diff = imf[:, :-1] - imf[:, 1:]
# print(np.std(diff, axis=0))

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
S_all_notch = copy.deepcopy(S_emg)

for IMF_arti in IMF_art:
    S_art_int = IMF_arti
    for i in range(1, 13):
        b, a = signal.iirnotch(i * f0, i * Q, fs)
        S_art_int = signal.lfilter(b, a, S_art_int)
    S_all_notch += S_art_int
    y.append(S_emg + S_art_int)
    SNR.append(np.sum(sEMG**2/(sEMG-y[-1])**2))

SNR = np.asarray(SNR)
imax = SNR.argmax()
cleaned_EMG = y[imax]

# Applying only the Notch filter
S_notch = sEMG
for i in range(1, 13):
    b, a = signal.iirnotch(i * f0, Q, fs)
    S_notch = signal.lfilter(b, a, S_notch)

# Applying only the comb filter
b, a = signal.iircomb(f0, Q, ftype='notch', fs=fs)
S_comb = signal.lfilter(b, a, sEMG)

plt.plot(time, sEMG, label="Raw data")
plt.plot(time, cleaned_EMG, label=f"EMD-Notch, Q={Q}")
plt.plot(time, S_emg, label="EMD + Th_sd")
plt.plot(time, S_all_notch, label="EMD + Th_sd + all Notch")
plt.plot(time, S_notch, label=f"Notch, Q={Q}")
plt.plot(time, S_comb, label=f"Comb filter, Q={Q}")
plt.legend()
plt.show()


# NRMSE and SNR
NRMSE = []
SNR = []

SD_EMG = np.std(sEMG)

for sig in [cleaned_EMG, S_emg, S_all_notch, S_notch, S_comb]:
    SNR.append(np.sum(sig ** 2 / (sEMG - sig) ** 2))
    NRMSE.append(np.sqrt(np.sum((sig - sEMG) ** 2) / len(sig)) / SD_EMG)

names = ["EMD-Notch", "EMD + Th_sd", "EMD + Th_sd + all Notch",
    "Notch", "Comb filter"]
plt.bar(names, NRMSE, label="NRMSE")
plt.xlabel("Filter")
plt.legend()

plt.figure()
plt.bar(names, SNR, label="SNR")
plt.xlabel("Filter")
plt.legend()
plt.show()