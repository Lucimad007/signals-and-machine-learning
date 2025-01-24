import os
from pydub import AudioSegment
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

folder_path = "26-hbd-notes" 

def print_dominant_frequency(file_path):
    audio = AudioSegment.from_file(file_path, format="m4a")

    samples = np.array(audio.get_array_of_samples())

    sample_rate = audio.frame_rate

    # normalize the signal
    samples = samples / np.max(np.abs(samples))

    # FFT
    n = len(samples)
    fft_values = fft(samples)
    frequencies = fftfreq(n, 1 / sample_rate)

    positive_freq = frequencies[:n // 2]
    positive_fft_values = np.abs(fft_values[:n // 2])

    # dominant frequency (frequency with the highest magnitude)
    dominant_frequency = positive_freq[np.argmax(positive_fft_values)]

    print(f"File: {os.path.basename(file_path)}")
    print(f"Dominant Frequency: {dominant_frequency:.2f} Hz")
    print("-" * 40)

# function to plot the frequencies of a single audio file
def analyze_audio(file_path):
    audio = AudioSegment.from_file(file_path, format="m4a")

    samples = np.array(audio.get_array_of_samples())

    sample_rate = audio.frame_rate

    # normalize the signal
    samples = samples / np.max(np.abs(samples))

    # FFT
    n = len(samples)
    fft_values = fft(samples)
    frequencies = fftfreq(n, 1 / sample_rate)

    positive_freq = frequencies[:n // 2]
    positive_fft_values = np.abs(fft_values[:n // 2])

    # plot the frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freq, positive_fft_values)
    plt.title(f"Frequency Spectrum: {os.path.basename(file_path)}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

for filename in os.listdir(folder_path):
    if filename.endswith(".m4a"):
        file_path = os.path.join(folder_path, filename)
        print(f"Analyzing: {filename}")
        print_dominant_frequency(file_path)