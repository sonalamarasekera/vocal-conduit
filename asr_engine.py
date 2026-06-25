import sherpa_onnx
import soundfile as sf
import time


class ASREngine:

    def __init__(self, model_dir, num_threads=2, sample_rate=16000):

        self.sample_rate = sample_rate

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
            model=f"{model_dir}/model.int8.onnx",
            tokens=f"{model_dir}/tokens.txt",
            num_threads=num_threads,
            sample_rate=sample_rate
        )

    def transcribe_wav(self, wav_path: str) -> str:

        audio, sr = sf.read(wav_path)

        #if sr != self.sample_rate:
        #    raise ValueError(f"Expected {self.sample_rate}, got {sr}")

        stream = self.recognizer.create_stream()
        stream.accept_waveform(sr, audio)

        start = time.time()
        self.recognizer.decode_stream(stream)
        latency = time.time() - start

        return stream.result.text, latency
