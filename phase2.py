import os
from pydub import AudioSegment
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def elbow_method():
    X_train = []  
    y_train = []  

    for note_name, frequencies in known_notes.items():
        for freq in frequencies:
            X_train.append(freq)
            y_train.append(note_name)

    X_train = np.array(X_train).reshape(-1, 1)  
    y_train = np.array(y_train) 

    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    error_rates = [] 
    k_values = range(1, 20)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_split, y_train_split)

        y_pred = knn.predict(X_val)

        error_rate = 1 - accuracy_score(y_val, y_pred)
        error_rates.append(error_rate)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, error_rates, marker="o")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Error Rate")
    plt.grid()
    plt.show()

    min_value = np.argmin(error_rates)

    indices = np.where(error_rates == min_value)[0]
    last_occurrence_index = indices[-1]
    best_k = k_values[last_occurrence_index]
    print(f"Best k value: {best_k}")

def get_dominant_frequency(file_path):
    audio = AudioSegment.from_file(file_path, format="m4a")
    samples = np.array(audio.get_array_of_samples())

    sample_rate = audio.frame_rate

    samples = samples / np.max(np.abs(samples))

    n = len(samples)
    fft_values = fft(samples)
    frequencies = fftfreq(n, 1 / sample_rate)

    positive_freq = frequencies[:n // 2]
    positive_fft_values = np.abs(fft_values[:n // 2])

    dominant_frequency = positive_freq[np.argmax(positive_fft_values)]

    return dominant_frequency, positive_freq, positive_fft_values

known_notes_folder = "knn/train" 
known_notes = {} 

for filename in os.listdir(known_notes_folder):
    if filename.endswith(".m4a"):
        note_name = filename.split("_")[0]

        file_path = os.path.join(known_notes_folder, filename)
        dominant_frequency, positive_freq, positive_fft_values = get_dominant_frequency(file_path)

        if note_name not in known_notes:
            known_notes[note_name] = []
        known_notes[note_name].append(dominant_frequency)

        # Plot FFT for each training file
        plt.figure(figsize=(10, 6))
        plt.plot(positive_freq, positive_fft_values)
        plt.title(f"FFT for {filename} (Dominant Frequency: {dominant_frequency:.2f} Hz)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid()
        plt.show()

X_train = []  
y_train = []  

for note_name, frequencies in known_notes.items():
    for freq in frequencies:
        X_train.append(freq)
        y_train.append(note_name)

X_train = np.array(X_train).reshape(-1, 1) 
y_train = np.array(y_train) 

knn = KNeighborsClassifier(n_neighbors=3)   
knn.fit(X_train, y_train)

unknown_notes_folder = "knn/test"  

for filename in os.listdir(unknown_notes_folder):
    if filename.endswith(".m4a"):
        file_path = os.path.join(unknown_notes_folder, filename)
        dominant_frequency, _, _ = get_dominant_frequency(file_path)

        predicted_note = knn.predict([[dominant_frequency]])

        print(f"File: {filename}")
        print(f"Dominant Frequency: {dominant_frequency:.2f} Hz")
        print(f"Predicted Note: {predicted_note[0]}")
        print("-" * 40)

elbow_method()

# bar plot for training data frequencies
plt.figure(figsize=(12, 6))
for note_name, frequencies in known_notes.items():
    plt.bar(note_name, np.mean(frequencies), label=note_name)
plt.title("Training Data Frequencies")
plt.xlabel("Note")
plt.ylabel("Frequency (Hz)")
plt.legend()
plt.show()

# bar plot for test data frequencies
test_frequencies = []
test_notes = []

for filename in os.listdir(unknown_notes_folder):
    if filename.endswith(".m4a"):
        file_path = os.path.join(unknown_notes_folder, filename)
        dominant_frequency, _, _ = get_dominant_frequency(file_path)
        predicted_note = knn.predict([[dominant_frequency]])[0]
        test_frequencies.append(dominant_frequency)
        test_notes.append(predicted_note)

plt.figure(figsize=(12, 6))
plt.bar(test_notes, test_frequencies)
plt.title("Test Data Frequencies")
plt.xlabel("Predicted Note")
plt.ylabel("Frequency (Hz)")
plt.show()