import soundfile as sf
import time
import os


def play_audio(samples, sample_rate, filename="tts_output.wav"):

    sf.write(filename, samples, sample_rate)

    print(f"[Audio] Saved to {os.path.abspath(filename)}")
