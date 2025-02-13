import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio


# Adjust the filepath to match the location of your audio files
filepath = r"D:\Transformation_fichier_audio\ressources\fichierwav\zouk.0000"
for i in range(2):
    # Load the audio file
    audio_path = filepath + str(i) + ".wav"
    audio, sfreq = librosa.load(audio_path)

    # Create a time axis
    time = np.arange(0, len(audio)) / sfreq

    # Plot the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, audio)
    plt.title(f"Audio Waveform - File {i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Sound Amplitude")
    plt.show()

    # Play the audio using IPython.display.Audio
    (Audio(audio_path)).autoplay