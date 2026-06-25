import sherpa_onnx
import time
import soundfile as sf
from fix_asr_text import fix_asr_text

wav_file = "/content/chinese_male.mp3"

audio, sample_rate = sf.read(wav_file)

print(f"Sample rate of test file: {sample_rate}")

start = time.time()

recognizer = sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
  model = "/content/sherpa-onnx-zh-wenet-wenetspeech/model.int8.onnx",
  tokens = "/content/sherpa-onnx-zh-wenet-wenetspeech/tokens.txt",
  num_threads = 2,
  sample_rate = 16000
)

stream = recognizer.create_stream()
stream.accept_waveform(sample_rate, audio)

recognizer.decode_stream(stream)

end = time.time()

asr_text = stream.result.text

llmstart = time.time()
fixed_text = fix_asr_text(asr_text)
llmstop = time.time()

print(f"\nASR only: {asr_text}")
print(f"LLM fixed = {fixed_text}")
print(f"\nASR Latency = {end - start} seconds")
print(f"LLM rescore latency = {llmstop - llmstart} seconds\n")