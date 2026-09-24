"""CPU fallback (faster-whisper / CTranslate2) for development machines without an NPU."""

from __future__ import annotations

import numpy as np

from netra.asr import Transcript


class CpuWhisper:
    def __init__(self, model_size: str = "base", language: str = "auto") -> None:
        from faster_whisper import WhisperModel

        try:  # stay offline once the model is cached
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8", local_files_only=True)
        except Exception:
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.language = None if language == "auto" else language
        self.engine = f"Whisper-{model_size} · CPU (faster-whisper, dev fallback)"

    def transcribe(self, audio: np.ndarray) -> Transcript:
        segments, info = self.model.transcribe(
            audio.astype(np.float32), language=self.language, beam_size=1, vad_filter=False
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return Transcript(text, info.language or "en")
