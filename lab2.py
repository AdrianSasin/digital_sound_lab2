import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

fs = 44100
sine_time = 1

#zad. 1.

def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val == min_val:
        return np.zeros_like(signal)
    normalized_signal = (signal - min_val) / (max_val - min_val)
    normalized_signal *= 2
    normalized_signal -= 1
    return normalized_signal

def generate_sine(freq, time, phase = 0):
    return np.array(np.sin(2 * np.pi * freq * np.arange(0,time, 1 / fs) - phase, dtype =np.float32))
f1 = 500
f2 = 1500
f3 = 3000

A1 = 1
A2 = 2
A3 = 0.5


sine1 = A1 * generate_sine(f1, sine_time )
sine2 = A2 * generate_sine(f2, sine_time )
sine3 = A3 * generate_sine(f3, sine_time )

sine_mix = sine1 + sine2 + sine3
sine_mix = normalize_signal(sine_mix)

sd.play(sine_mix, fs)
sd.wait()

#zad 2.

def get_spectrum(signal):
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(fft_result), 1/fs)
    n = len(fft_result) // 2
    return fft_freq, fft_result, n

fft_freq1, fft_result1, n = get_spectrum(sine_mix)
plt.plot(fft_freq1[:n], np.abs(fft_result1[:n]) * 2 / len(sine_mix))
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.show()