import os
import numpy as np
from scipy.fft import fft
from pydub import AudioSegment

def dominant_frequency(sound):
    data = np.array(sound.get_array_of_samples())
    rate = sound.frame_rate
    size = len(data)
    transform = fft(data)
    freq = np.arange(size) * (rate / size)
    amp = np.abs(transform)
    peak = np.argmax(amp)
    return freq[peak]

def load_and_process_files(folder, file_list):
    processed_data = []
    for file in file_list:
        path = os.path.join(folder, file)
        sound = AudioSegment.from_file(path, format="m4a")
        freq = dominant_frequency(sound)
        processed_data.append([file, freq])
    return processed_data

def classify_note(file, ref_data):
    path = os.path.join(dir_b, file)
    sound = AudioSegment.from_file(path, format="m4a")
    freq = dominant_frequency(sound)
    length = len(sound.get_array_of_samples()) / sound.frame_rate

    diff = [abs(entry[1] - freq) for entry in ref_data]
    closest = np.argmin(diff)
    matched = ref_data[closest][0]

    # range of durations based on all files
    if length < 0.3:
        note_type = '8th'
    elif 0.3 <= length <= 0.6:
        note_type = '4th'
    else:
        note_type = 'dot4th'

    return [file, matched, length, note_type]

def create_final_song(sequence, output_data, output_folder):
    final_sound = AudioSegment.empty()
    for note_name, type_n in sequence:
        for entry in output_data:
            if entry[1] == note_name and entry[3] == type_n:
                path = os.path.join(output_folder, entry[0])
                sound = AudioSegment.from_file(path, format="m4a")
                final_sound += sound
                break
    return final_sound

dir_a = '8-named-notes'
dir_b = '26-hbd-notes'

list_a = [f for f in os.listdir(dir_a) if f.endswith('.m4a')]
list_b = [f for f in os.listdir(dir_b) if f.endswith('.m4a')]

ref_data = load_and_process_files(dir_a, list_a)

output_data = [classify_note(file, ref_data) for file in list_b]

# HBD
sequence = [
    ('Sol_octave1.m4a', '8th'),
    ('Sol_octave1.m4a', '4th'),
    ('Do_octave2.m4a', 'dot4th'),
    ('Sol_octave1.m4a', '8th'),
    ('Sol_octave1.m4a', '4th'),
    ('Do_octave2.m4a', 'dot4th'),
    ('Sol_octave1.m4a', '8th'),
    ('Sol_octave1.m4a', '4th'),
    ('Do_octave2.m4a', '8th'),
    ('Do_octave2.m4a', '4th'),
    ('Si_octave1.m4a', '8th'),
    ('La_octave1.m4a', '4th'),
    ('Si_octave1.m4a', 'dot4th'),
    ('Sol_octave1.m4a', '8th'),
    ('Sol_octave1.m4a', '4th'),
    ('Si_octave1.m4a', '4th'),
    ('Sol_octave1.m4a', '8th'),
    ('Sol_octave1.m4a', '4th'),
    ('Si_octave1.m4a', 'dot4th'),
    ('Sol_octave1.m4a', '8th'),
    ('Sol_octave1.m4a', '4th'),
    ('La_octave1.m4a', '8th'),
    ('Sol_octave1.m4a', '4th'),
    ('La_octave1.m4a', '8th'),
    ('Si_octave1.m4a', '4th'),
    ('Do_octave2.m4a', 'dot4th')
]

final_sound = create_final_song(sequence, output_data, dir_b)

final_sound.export('HBD.m4a', format="mp4", codec="aac")    # the output will be m4a
print("Song created successfully.")