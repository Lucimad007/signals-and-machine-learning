import os
from pydub import AudioSegment
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


folder_path = '8-named-notes'
folder_path2 = "26-hbd-notes" 

def get_positive_frequencies(file_path):
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

    return positive_freq, positive_fft_values

def get_dominant_frequency(file_path):
    positive_freq, positive_fft_values = get_positive_frequencies(file_path)

    dominant_frequency = positive_freq[np.argmax(positive_fft_values)]

    return dominant_frequency

def print_dominant_frequency(file_path):
    dominant_frequency = get_dominant_frequency(file_path)

    print(f"File: {os.path.basename(file_path)}")
    print(f"Dominant Frequency: {dominant_frequency:.2f} Hz")
    print("-" * 40)

# function to plot the frequencies of a single audio file
def analyze_audio(file_path):
    
    positive_freq, positive_fft_values = get_positive_frequencies

    # plot the frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freq, positive_fft_values)
    plt.title(f"Frequency Spectrum: {os.path.basename(file_path)}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

def compare_dominant_frequencies(file1, file2, tolerance=100):
    dominant_freq1 = get_dominant_frequency(file1)
    dominant_freq2 = get_dominant_frequency(file2)

    if np.abs(dominant_freq1 - dominant_freq2) <= tolerance:
        return True
    else:
        return False

file1 = '8-named-notes/Do_octave1.m4a'

for filename in os.listdir(folder_path2):
    if filename.endswith(".m4a"):
        file_path = os.path.join(folder_path2, filename)
        print(f"Analyzing: {filename}")
        print_dominant_frequency(file_path)
        print(compare_dominant_frequencies(file1, file_path))