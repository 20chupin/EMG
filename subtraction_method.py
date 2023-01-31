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

data = Analogs.from_c3d(data_path, usecols=channels)

time = data.time[50500:61000:5]
sEMG = np.asarray(data.sel(channel="Sensor 13.IM EMG13")[50500:61000:5])

class PeakDetection():

    def __init__(self, time, data, fs, T_window):
        self.time = time
        self.data = data
        self.T_window = T_window
        self.fs =fs
        self.threshold = 1e-4

    def moving_average_filtering(self, data, T):
        av_data = np.zeros(len(data))
        for i in range(len(data)):
            start = max(0, int(i-fs*T))
            end = min(len(data), int(i+fs*T))
            av_data[i] = np.mean(data[start:end])
        return av_data

    def ups_and_downs(self):
        time = np.asarray(self.time)
        self.rect_data = self.moving_average_filtering(np.abs(self.data), 0.01)
        self.rect_data = self.moving_average_filtering(self.rect_data, 0.005)
        b, a = signal.butter(2, 20, btype='highpass', fs=fs)
        self.rect_data = signal.lfilter(b, a, self.rect_data)
        i = 0
        self.ups_and_downs = {
            "up": {
                "time": [],
                "i": []
            },
            "down": {
                "time": [],
                "i": []
            }
        }
        look_for_up = True
        while i < len(self.rect_data):
            if look_for_up:
                if self.rect_data[i] < self.threshold:
                    i += 1
                else:
                    self.ups_and_downs["up"]["i"].append(i)
                    self.ups_and_downs["up"]["time"].append(time[i])
                    i += 1
                    look_for_up = False
            else:
                if self.rect_data[i] > -self.threshold:
                    i += 1
                else:
                    self.ups_and_downs["down"]["i"].append(i)
                    self.ups_and_downs["down"]["time"].append(time[i])
                    look_for_up = True
        return self.ups_and_downs

    def local_min_and_max(self):
        self.min_and_max = {
            "min":{
                "time": [],
                "i": [],
                "value": []
            },
            "max":{
                "time": [],
                "i": [],
                "value": []
            }
        }
        for i in range(len(self.ups_and_downs["up"]["i"])-1):
            up = self.ups_and_downs["up"]["i"][i]
            down = self.ups_and_downs["down"]["i"][i]
            self.min_and_max["min"]["value"].append(np.min(self.data[up:down]))
            self.min_and_max["min"]["time"].append(time[up + np.argmin(self.data[up:down])])
            self.min_and_max["min"]["i"].append(up + np.argmin(self.data[up:down]))
            self.min_and_max["max"]["value"].append(np.max(self.data[up:down]))
            self.min_and_max["max"]["time"].append(time[up + np.argmax(self.data[up:down])])
            self.min_and_max["max"]["i"].append(up + np.argmax(self.data[up:down]))
        return self.min_and_max

class SubstractionMethod(PeakDetection):
    def __init__(self, time, data, fs, T_window):
        super().__init__(time, data, fs, T_window)
    
    def mean_peak(self):
        self.peaks = []
        window = int(self.T_window * self.fs)
        for i in self.min_and_max["max"]["i"]:
            self.peaks.append(self.data[i - window : i + window]/self.data[i])
        self.template = np.mean(self.peaks, axis=0)
        self.std = np.std(self.peaks, axis=0)
        return self.template

    def subtraction(self):
        self.cleaned_EMG = copy.deepcopy(self.data)
        self.sub= np.zeros(len(self.data))
        window = int(self.T_window * self.fs)
        for i in self.min_and_max["max"]["i"]:
            self.sub[i - window : i + window] = self.template * self.cleaned_EMG[i]
            self.cleaned_EMG[i - window : i + window] = self.cleaned_EMG[i - window : i + window] - self.template * self.cleaned_EMG[i]
        return self.cleaned_EMG

SM = SubstractionMethod(time, sEMG, fs, 0.005)
SM.ups_and_downs()
SM.local_min_and_max()
SM.mean_peak()
SM.subtraction()

fig1, axs1 = plt.subplots(4, 1, sharex=True, sharey=True)
for i in [0, 1, 3]:
    axs1[i].plot(time, sEMG, label="sEMG")

for i in range(4):
    axs1[i].set_ylabel("EMG (V)")

axs1[3].set_xlabel("Time (s)")

axs1[0].plot(time, 10*SM.rect_data, label="Averaged and filtered sEMG")
axs1[0].plot(time, 10*SM.threshold*np.ones(len(time)), "r", label="Positive threshold")
axs1[0].plot(time, -10*SM.threshold*np.ones(len(time)), "g", label="Negative threshold")
axs1[0].plot(SM.ups_and_downs["up"]["time"], np.zeros(len(SM.ups_and_downs["up"]["time"])), 'r+', label="Starts of the artifacts")
axs1[0].plot(SM.ups_and_downs["down"]["time"], np.zeros(len(SM.ups_and_downs["down"]["time"])), 'g+', label="Ends of the artifacts")
axs1[0].legend()
axs1[0].set_title("Peaks detection")

axs1[1].plot(SM.min_and_max["min"]["time"], SM.min_and_max["min"]["value"], '+', label="Peaks' min")
axs1[1].plot(SM.min_and_max["max"]["time"], SM.min_and_max["max"]["value"], '+', label="Peaks' max")
axs1[1].legend()
axs1[1].set_title("Peaks' min and max detection")

plt.figure(1)
axs1[2].plot(time, SM.sub, "k", label="Signal to be subtracted")
axs1[2].legend()
axs1[2].set_title("Signal to be subtracted calculated from the mean of the peaks (see 'Creation of anormalized peak template')")

axs1[3].plot(time, SM.cleaned_EMG, label="Cleaned EMG")
axs1[3].legend()
axs1[3].set_title("Cleaned EMG")

fig1.suptitle("A subtration method for supressing the EMG's artifacts")

fig2, axs2 = plt.subplots(1, 2, sharex=True, sharey=True)
for i in range(len(SM.peaks)):
    axs2[0].plot(SM.peaks[i])
axs2[0].set_title("Normalized peaks")

axs2[1].plot(SM.template)
axs2[1].fill_between(np.arange(len(SM.template)), SM.template - SM.std, SM.template + SM.std, alpha=0.2)
axs2[1].set_title("Normalized peak template")

axs2[0].set_ylabel("EMG (V)")

fig2.suptitle("Creation of a normalized peak template from the mean of normalized peaks")

plt.show()
