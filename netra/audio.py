"""Microphone capture with automatic end-of-speech detection, plus short "earcons" so a
blind user always knows whether Netra is listening, thinking or has hit an error."""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 30


def _tone(freqs: list[float], dur: float = 0.09, vol: float = 0.18, sr: int = 22050) -> np.ndarray:
    parts = []
    for f in freqs:
        t = np.arange(int(sr * dur)) / sr
        env = np.minimum(1, np.minimum(t, dur - t) / 0.012)  # click-free fade in/out
        parts.append(vol * env * np.sin(2 * np.pi * f * t))
    return np.concatenate(parts).astype(np.float32)


EARCONS = {
    "listen": _tone([660, 990]),  # rising: go ahead and speak
    "think": _tone([520], dur=0.07),  # got it, working
    "error": _tone([330, 220], dur=0.14),  # falling: something went wrong
}


def earcon(name: str) -> None:
    try:
        import sounddevice as sd

        sd.play(EARCONS[name], 22050)
    except Exception:
        pass  # no audio device: stay silent rather than crash


def record_until_silence(
    silence_s: float = 1.1,
    max_s: float = 15.0,
    no_speech_timeout_s: float = 5.0,
    stop_event: threading.Event | None = None,
) -> np.ndarray:
    """Record mono 16 kHz audio until the speaker pauses, `max_s` passes, or `stop_event` is set.

    Energy-based endpointing with an adaptive noise floor measured in the first 300 ms.
    Returns an empty array if nobody spoke.
    """
    import sounddevice as sd

    frame = SAMPLE_RATE * FRAME_MS // 1000
    chunks: list[np.ndarray] = []
    noise: list[float] = []
    speech_started = False
    last_voice = time.monotonic()
    t0 = time.monotonic()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=frame) as stream:
        while True:
            block, _ = stream.read(frame)
            block = block[:, 0].copy()
            chunks.append(block)
            rms = float(np.sqrt(np.mean(block**2)) + 1e-9)
            now = time.monotonic()
            if now - t0 < 0.3:
                noise.append(rms)
                continue
            threshold = max(0.012, 3.0 * float(np.median(noise)))
            if rms > threshold:
                speech_started = True
                last_voice = now
            if stop_event is not None and stop_event.is_set():
                break
            if speech_started and now - last_voice >= silence_s:
                break
            if not speech_started and now - t0 >= no_speech_timeout_s:
                return np.zeros(0, dtype=np.float32)
            if now - t0 >= max_s:
                break
    return np.concatenate(chunks) if speech_started else np.zeros(0, dtype=np.float32)


def load_wav(path: str | Path) -> np.ndarray:
    """Read a WAV file as mono float32 at 16 kHz."""
    from scipy.signal import resample_poly

    with wave.open(str(path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        audio = (audio - 128) / 128
    else:
        audio /= float(np.iinfo(dtype).max)
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sr).astype(np.float32)
    return audio
