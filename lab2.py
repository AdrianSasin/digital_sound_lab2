import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile



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
sine_mix_normalized = normalize_signal(sine_mix)

sd.play(sine_mix_normalized, fs)
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

#zad. 3.

T = 1 / f1
periods = 5

samples = int(periods * T * fs)

plt.plot(np.arange(samples) / fs, sine_mix[:samples])
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("First periods of signal")
plt.grid(True)
plt.show()

#zad. 4.

left = generate_sine(f2, sine_time )
right = generate_sine(f3, sine_time )
stereo_sig = np.column_stack((left, right))
left_channel = stereo_sig[:, 0]
right_channel = stereo_sig[:, 1]

sd.play(stereo_sig, fs)
sd.wait()

duration = 0.005  # 5 ms
samples = int(duration * fs)

time_vec = np.arange(samples) / fs

plt.subplot(2, 1, 1)
plt.plot(time_vec, left_channel[:samples])
plt.title(f"Left channel ({f2} Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(time_vec, right_channel[:samples])
plt.title(f"Right channel ({f3} Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

#zad.5.

f_stereo = 500

A_left = 1.0
A_right = 0.3

left_channel = A_left * generate_sine(f_stereo, sine_time)
right_channel = A_right * generate_sine(f_stereo, sine_time)

stereo_sig = np.column_stack((left_channel, right_channel))

sd.play(stereo_sig, fs)
sd.wait()

duration = 0.01
samples = int(duration * fs)

time_vec = np.arange(samples) / fs

plt.subplot(2, 1, 1)
plt.plot(time_vec, left_channel[:samples])
plt.title(f"Left channel ({f_stereo} Hz, A={A_left})")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(time_vec, right_channel[:samples])
plt.title(f"Right channel ({f_stereo} Hz, A={A_right})")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

#zad. 6.

samplerate, audio = wavfile.read("file_example_WAV_1MG.wav")
audio = normalize_signal(audio)

sd.play(audio, samplerate)
sd.wait()

A1 = 0.1
A2 = 0.5

audio1 = A1*audio
audio2 = A2*audio

#odegranie dźwięków
sd.play(audio1, samplerate)
sd.wait()
sd.play(audio2, samplerate)
sd.wait()

samples = len(audio)
time_vec = np.arange(samples) / samplerate

plt.subplot(2, 1, 1)
plt.plot(time_vec, audio1)
plt.title("Audio A1 = 0.1")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.ylim([-1, 1])

plt.subplot(2, 1, 2)
plt.plot(time_vec, audio2)
plt.title("Audio A2 = 0.5")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.ylim([-1, 1])

plt.tight_layout()
plt.show()

#zad. 7.

samplerate, audio = wavfile.read("file_example_WAV_1MG.wav")

if audio.ndim == 2:
    audio = audio[:, 0]

audio = normalize_signal(audio)

audio_reversed = audio[::-1]

sd.play(audio, samplerate)
sd.wait()

sd.play(audio_reversed, samplerate)
sd.wait()


samples = len(audio)
time_vec = np.arange(samples) / samplerate

plt.subplot(2, 1, 1)
plt.plot(time_vec, audio)
plt.title("Original signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(time_vec, audio_reversed)
plt.title("Reversed signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

#zad. 8.

def new_get_spectrum(signal, fs):
    fft_result = np.fft.rfft(signal)
    fft_freq = np.fft.rfftfreq(len(signal), 1 / fs)
    return fft_freq, fft_result

# usunięcie składowej stałej
audio_no_dc = audio - np.mean(audio)
audio_reversed_no_dc = audio_reversed - np.mean(audio_reversed)

# widmo sygnału oryginalnego
fft_freq1, fft_result1 = new_get_spectrum(audio_no_dc, samplerate)

# widmo sygnału odwróconego
fft_freq2, fft_result2 = new_get_spectrum(audio_reversed_no_dc, samplerate)

plt.subplot(2, 1, 1)
plt.plot(fft_freq1, np.abs(fft_result1) * 2 / len(audio_no_dc))
plt.title("Spectrum of original signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(fft_freq2, np.abs(fft_result2) * 2 / len(audio_reversed_no_dc))
plt.title("Spectrum of reversed signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()