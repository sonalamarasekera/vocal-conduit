import numpy as np
import soundfile as sf
from tts_sherpa import SherpaTTSEngine

tts = SherpaTTSEngine("/content/sherpa-onnx-vits-zh-ll")

samples, sr = tts.synthesize("你好，这是一个语音识别的项目")

samples = np.asarray(samples, dtype=np.float32)

sf.write("tts_test.wav", samples, sr)

print("Saved tts_test.wav")
