from mutagen.mp4 import MP4

def get_m4a_duration(file_path):
    audio = MP4(file_path)
    duration = audio.info.length
    return duration

file_path = '8-named-notes/Sol_octave1.m4a'
duration = get_m4a_duration(file_path)
print(f"Duration: {duration} seconds")
import glob
import os

current_dir = os.getcwd()
m4a_files = glob.glob(os.path.join(current_dir + '/8-named-notes', '*.m4a'))

# Print the list of .m4a files
for file_path in m4a_files:
    print(f"{os.path.basename(file_path)}")
