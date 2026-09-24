"""Speech-to-text. On Snapdragon, Whisper runs on the Hexagon NPU through ONNX Runtime's
QNN execution provider (AI Hub precompiled encoder/decoder). Elsewhere it falls back to
faster-whisper on the CPU so the rest of the app can be developed on any laptop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from netra import device
from netra.config import AsrConfig

log = logging.getLogger("netra")
SAMPLE_RATE = 16000


@dataclass
class Transcript:
    text: str
    language: str  # ISO code, e.g. "en" / "hi"


class ASR(Protocol):
    engine: str

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """audio: mono float32 at 16 kHz."""
        ...


def load_asr(cfg: AsrConfig) -> ASR:
    want = cfg.backend
    model_dir = Path(cfg.model_dir)
    has_npu_model = (model_dir / "encoder.onnx").is_file() and (model_dir / "decoder.onnx").is_file()
    if want in ("auto", "npu"):
        ok, why = device.npu_status()
        if ok and has_npu_model:
            from netra.asr.qnn_whisper import QnnWhisper

            return QnnWhisper(model_dir, cfg.model_size, cfg.language)
        if want == "npu":
            raise RuntimeError(
                f"NPU ASR requested but unavailable: {why}"
                + ("" if has_npu_model else f"; no encoder/decoder.onnx in {model_dir}")
            )
        log.warning("ASR falling back to CPU: %s", why if not ok else f"no NPU model in {model_dir}")
    from netra.asr.cpu_whisper import CpuWhisper

    return CpuWhisper(cfg.model_size, cfg.language)
