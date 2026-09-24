"""Whisper on the Snapdragon Hexagon NPU.

Runs the Qualcomm AI Hub `whisper_*` precompiled_qnn_onnx export (encoder.onnx +
decoder.onnx) with ONNX Runtime's QNN EP. The static-shape KV-cache decode loop is
adapted from Qualcomm's ai-hub-apps `whisper_windows_py` sample (BSD-3-Clause),
extended with forced language prompting (Hindi / English) and language detection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from netra.asr import SAMPLE_RATE, Transcript
from netra.device import open_npu_session

MEAN_DECODE_LEN = 200  # static self-attention KV length baked into the AI Hub export
MASK_NEG = -100.0
CHUNK_SECONDS = 30

_NP_TYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint16)": np.uint16,
    "tensor(uint8)": np.uint8,
}


def _run(session, *args: np.ndarray) -> tuple[np.ndarray, ...]:
    feed = {
        i.name: a.astype(_NP_TYPES[i.type], copy=False)
        for i, a in zip(session.get_inputs(), args, strict=False)
    }
    return tuple(session.run([o.name for o in session.get_outputs()], feed))


class QnnWhisper:
    def __init__(self, model_dir: Path, model_size: str = "base", language: str = "auto") -> None:
        from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperTokenizer

        hf_id = f"openai/whisper-{model_size}"
        self.config = WhisperConfig.from_pretrained(hf_id)
        self.tokenizer = WhisperTokenizer.from_pretrained(hf_id)
        self.features = WhisperFeatureExtractor.from_pretrained(hf_id)
        self.encoder = open_npu_session(str(model_dir / "encoder.onnx"))
        self.decoder = open_npu_session(str(model_dir / "decoder.onnx"))
        self.language = language
        self.engine = f"Whisper-{model_size} · Hexagon NPU (ORT QNN EP)"
        tok = self.tokenizer.convert_tokens_to_ids
        self._lang_ids = {tok(f"<|{code}|>"): code for code in ("en", "hi", "mr", "bn", "ta", "te", "gu", "ur")}
        self._transcribe_id = tok("<|transcribe|>")
        self._no_ts_id = tok("<|notimestamps|>")

    def transcribe(self, audio: np.ndarray) -> Transcript:
        texts, langs = [], []
        for start in range(0, max(len(audio), 1), CHUNK_SECONDS * SAMPLE_RATE):
            ids = self._decode(audio[start : start + CHUNK_SECONDS * SAMPLE_RATE])
            langs += [self._lang_ids[i] for i in ids if i in self._lang_ids]
            texts.append(self.tokenizer.decode(ids, skip_special_tokens=True).strip())
        lang = langs[0] if langs else (self.language if self.language != "auto" else "en")
        return Transcript(" ".join(t for t in texts if t), lang)

    def _prompt(self) -> list[int]:
        sot = self.config.decoder_start_token_id
        if self.language == "auto":
            return [sot]  # let the decoder emit its own language token
        lang_id = self.tokenizer.convert_tokens_to_ids(f"<|{self.language}|>")
        return [sot, lang_id, self._transcribe_id, self._no_ts_id]

    def _decode(self, audio: np.ndarray) -> list[int]:
        cfg = self.config
        feats = self.features(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")["input_features"]
        cross = _run(self.encoder, feats)  # flat (k0, v0, k1, v1, ...) cross-attention cache

        heads, dim = cfg.decoder_attention_heads, cfg.d_model
        k_self = np.zeros((heads, 1, dim // heads, MEAN_DECODE_LEN - 1), dtype=np.float32)
        v_self = np.zeros((heads, 1, MEAN_DECODE_LEN - 1, dim // heads), dtype=np.float32)
        self_cache: tuple[np.ndarray, ...] = (k_self, v_self) * cfg.decoder_layers

        prompt = self._prompt()
        out = list(prompt)
        mask = np.full((1, 1, 1, MEAN_DECODE_LEN), MASK_NEG, dtype=np.float32)
        position = np.array([0], dtype=np.int32)
        for n in range(MEAN_DECODE_LEN - 1):
            mask[:, :, :, MEAN_DECODE_LEN - n - 1] = 0.0
            token = np.array([[out[n]]], dtype=np.int32)
            logits, *self_cache = _run(self.decoder, token, mask, *self_cache, *cross, position)
            position += 1
            if n < len(prompt) - 1:
                continue  # still teacher-forcing the prompt tokens
            nxt = int(np.argmax(logits))
            out.append(nxt)
            if nxt == cfg.eos_token_id:
                break
        return out
