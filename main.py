from asr_engine import ASREngine
from tts_sherpa import SherpaTTSEngine
from pipeline_controller import SpeechPipeline


asr = ASREngine(
    model_dir="/content/sherpa-onnx-zh-wenet-wenetspeech"
)

tts = SherpaTTSEngine(
    model_dir="/content/sherpa-onnx-vits-zh-ll"
)

pipeline = SpeechPipeline(asr, tts)

pipeline.run_from_wav("/content/chinese_male.mp3")
