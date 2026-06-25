#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import soundfile as sf
import librosa
from tqdm import tqdm
import sherpa_onnx


# ---------------- CONFIG ----------------
MODEL_PATH = "sherpa-onnx-zh-wenet-wenetspeech/model.int8.onnx"
TOKENS     = "sherpa-onnx-zh-wenet-wenetspeech/tokens.txt"

WAV_ROOT   = "data_aishell/wav_extracted"
OUT_FILE   = "data_aishell/transcript/asr_test_hyp.txt"

TARGET_SR  = 16000
# ----------------------------------------


def load_audio(path):
    audio, sr = sf.read(path)

    if len(audio.shape) > 1:
        audio = audio[:, 0]  # mono

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    return audio.astype("float32")


def main():

    print("Loading Sherpa-ONNX model...")

    recognizer = sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
        model=MODEL_PATH,
        tokens=TOKENS,
        num_threads=4,
        sample_rate=16000,
        feature_dim=80,
    )

    wav_files = sorted(glob.glob(f"{WAV_ROOT}/**/*.wav", recursive=True))
    print(f"Total wav files: {len(wav_files)}")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    with open(OUT_FILE, "w", encoding="utf-8") as fout:

        for wav in tqdm(wav_files):
            utt_id = os.path.basename(wav).replace(".wav", "")

            audio = load_audio(wav)

            stream = recognizer.create_stream()
            stream.accept_waveform(TARGET_SR, audio)
            recognizer.decode_stream(stream)

            text = stream.result.text.strip()

            fout.write(f"{utt_id} {text}\n")

    print("\nASR batch decoding finished.")
    print(f"Hypothesis saved to: {OUT_FILE}")


if __name__ == "__main__":
    main()
