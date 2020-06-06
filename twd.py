# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 11:25:14 2020

@author: Gaurav

for pydub to work following files from FFMEG are needed to be added in the working directory:
ffmpeg.exe
ffplay.exe
ffprobe.exe


"""

from scipy.signal import spectrogram
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pydub import AudioSegment
#from Ipython.display import Audio

#%%
fs = 100000

audio_negative = AudioSegment.from_file('negative.m4a')
audio_negative = np.array(audio_negative.get_array_of_samples())[:fs*10]
audio_positive = AudioSegment.from_file('tensorflow.m4a')
audio_positive = np.array(audio_positive.get_array_of_samples())[:fs*10]

f_neg, t_neg, Sxx_neg = spectrogram(audio_negative, fs=fs, window=('tukey', 1), nperseg=1000)
f_pos, t_pos, Sxx_pos = spectrogram(audio_positive, fs=fs, window=('tukey', 1), nperseg=1000)
Sxx_neg = np.log(Sxx_neg+0.00001)
Sxx_pos = np.log(Sxx_pos+0.00001)

y_pos = np.zeros((103, 1))
y_neg = np.zeros((103, 1))

for i in [0.9, 1.9, 2.9, 3.9, 5, 7.0, 8.1, 9.5, 11.2]:
    y_pos[int(103*i/10): int(103*(i+0.3)/10)] = 1

Sxx = np.concatenate([Sxx_neg[None,:], Sxx_pos[None]], axis=0).swapaxes(1,2)
y = np.concatenate([y_neg[None,:], y_pos[None,:]], axis=0)

del Sxx_neg, Sxx_pos, audio_negative, audio_positive, i, y_neg, y_pos, fs

fig = plt.figure()
ax = fig.add_subplot(221)
ax.pcolormesh(t_neg, f_neg, Sxx[0,:].T)
ax.set(xlabel='time', ylabel='freq', title = 'negative')
ax = fig.add_subplot(222)
ax.pcolormesh(t_pos, f_pos, Sxx[1,:].T)
ax.set(xlabel='time', ylabel='freq', title = 'positive')
ax = fig.add_subplot(223)
plt.plot(np.arange(103), y[0,:])
plt.xlim([0,103])
ax = fig.add_subplot(224)
plt.plot(np.arange(103), y[1,:])
plt.xlim([0,103])

#%%
input_shape = Sxx.shape[1], Sxx.shape[2]
from tensorflow.keras.layers import Dense, BatchNormalization, GRU, Conv1D, TimeDistributed, Activation, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
input_X = Input(shape=input_shape)
X = Conv1D(100, 10, 11)(input_X)

X = BatchNormalization()(X)
X = Activation('relu')(X)
X = Dropout(0.8)(X)

X = GRU(90, return_sequences=True)(X)
X = Dropout(0.8)(X)
X = BatchNormalization()(X)

X = GRU(90, return_sequences=True)(X)
X = Dropout(0.8)(X)
X = BatchNormalization()(X)
X = Dropout(0.8)(X)

X = TimeDistributed(Dense(1))(X)
X = Activation('sigmoid')(X)

model = Model(input_X, X)
model.summary()

model.compile(optimizer='adam' , loss='binary_crossentropy')
model.fit(Sxx,y, epochs=2000)

#%%

fs = 100000

audio_test = AudioSegment.from_file('test.m4a')
audio_test = np.array(audio_test.get_array_of_samples())[:fs*10]
f_neg, t_neg, Sxx_test = spectrogram(audio_test, fs=fs, window=('tukey', 1), nperseg=1000)
Sxx_test = np.log(Sxx_test+0.00001)[None,:].swapaxes(1,2)
y_pred = model.predict(Sxx_test)

fig = plt.figure()
ax = fig.add_subplot(221)
ax.pcolormesh(t_neg, f_neg, Sxx_test[0,:].T)
ax.set(xlabel='time', ylabel='freq', title = 'negative')
ax = fig.add_subplot(222)
ax.pcolormesh(t_pos, f_pos, Sxx_test[0,:].T)
ax.set(xlabel='time', ylabel='freq', title = 'positive')
ax = fig.add_subplot(223)
plt.plot(np.arange(103), y_pred[0,:])
plt.xlim([0,103])
ax = fig.add_subplot(224)
plt.plot(np.arange(103), y_pred[0,:])
plt.xlim([0,103])

#%% following codes give hint on how spectrograms work

def sin_generator(t, frequency=1, amplitude=1, start_time=0, end_time=3600):
    base_signal = amplitude * np.sin(t*(2*np.pi)*frequency)
    chopped_signal = np.zeros_like(base_signal)
    t_start = np.argmin(np.abs(t-start_time))
    t_final = np.argmin(np.abs(t-end_time))
    chopped_signal[t_start:t_final] = base_signal[t_start:t_final]
    return chopped_signal

total_time = 60* 60
fs = 1000
t = np.linspace(0, total_time, num=total_time*fs)

sin1 = sin_generator(t, frequency=1, amplitude=1, start_time=900, end_time=3600)
sin2 = sin_generator(t, frequency=50, amplitude=1, start_time=1800, end_time=3600)
sin3 = sin_generator(t, frequency=300, amplitude=0.01, start_time=2700, end_time=3600)
sin4 = sin_generator(t, frequency=10, amplitude=4, start_time=100, end_time=200)
sin5 = sin_generator(t, frequency=90, amplitude=6, start_time=500, end_time=700)
signal = sin1 + sin2 + sin3 + sin4 + sin5

del sin1, sin2, sin3, sin4, sin5, t

nperseg = 1000
f, t1, Sxx = spectrogram(signal, fs=fs, window=('tukey', 1), nperseg=nperseg)

plt.figure()
plt.pcolormesh(t1, f, np.log(Sxx), cmap='jet')
plt.xlabel('time')
plt.ylabel('freq')
plt.colorbar()