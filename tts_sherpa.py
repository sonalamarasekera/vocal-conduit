import sherpa_onnx
import numpy as np
import time

class SherpaTTSEngine:

    def __init__(self, model_dir):

        vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=f"{model_dir}/model.onnx",
            tokens=f"{model_dir}/tokens.txt",
            lexicon=f"{model_dir}/lexicon.txt",
            dict_dir=f"{model_dir}/dict"
        )

        model_config = sherpa_onnx.OfflineTtsModelConfig(
            vits=vits_config
        )

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=model_config,
            rule_fsts=f"{model_dir}/phone.fst"   # ← THIS IS THE KEY FIX
        )

        self.tts = sherpa_onnx.OfflineTts(tts_config)
    
    def synthesize(self, text, speaker_id=0, speed=1.0):
      start = time.time()
      audio = self.tts.generate(
      text=text,
      sid=speaker_id,
      speed=speed
      )

      samples = np.array(audio.samples, dtype=np.float32)
      latency = time.time() - start
      return audio.samples, audio.sample_rate, latency