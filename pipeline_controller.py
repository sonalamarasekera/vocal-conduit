from fix_asr_text import fix_asr_text
from audio_io import play_audio


class SpeechPipeline:

    def __init__(self, asr_engine, tts_engine):
        self.asr = asr_engine
        self.tts = tts_engine

    def run_from_wav(self, wav_path: str):

        # ---- ASR ----
        raw_text, asr_latency = self.asr.transcribe_wav(wav_path)
        print(f"[ASR] {raw_text}")
        print(f"[ASR latency] {asr_latency:.3f}s")

        # ---- LLM FIX ----
        fixed_text, llm_latency = fix_asr_text(raw_text)
        print(f"[LLM] {fixed_text}")
        print(f"[LLM Latency] {llm_latency:.3f}s")

        # ---- TTS ----
        samples, sr, tts_latency = self.tts.synthesize(fixed_text)
        print(f"[TTS Latency] {tts_latency:.3f}s")

        # ---- Total Latency ----
        print(f"[Total Latency] {(asr_latency)+(llm_latency)+(tts_latency):.3f}s")

        # ---- Playback ----
        play_audio(samples, sr)
