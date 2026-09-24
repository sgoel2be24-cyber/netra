"""Offline, interruptible text-to-speech.

Sentences are queued as the model streams them, so Netra starts talking while the rest
of the answer is still being generated. Backends:
  Windows: OneCore neural voices via WinRT (Hindi "Kalpana"/"Hemant" when the language
           pack is installed), falling back to SAPI through PowerShell.
  macOS:   the built-in `say` command (Hindi voice "Lekha").
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import subprocess
import sys
import threading

from netra.config import SpeechConfig

log = logging.getLogger("netra")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+|\n+")


def detect_language(text: str) -> str:
    return "hi" if _DEVANAGARI.search(text) else "en"


class SentenceBuffer:
    """Turns a stream of text deltas into speakable sentences."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        parts = _SENTENCE_END.split(self._buf)
        self._buf = parts.pop()  # the last piece may be an unfinished sentence
        return [p.strip() for p in parts if p.strip()]

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []


def clean_for_speech(text: str) -> str:
    text = re.sub(r"[*_#`>|]+", " ", text)  # stray markdown the model may emit
    text = re.sub(r"https?://\S+", "a link", text)
    return re.sub(r"\s+", " ", text).strip()


class Speaker:
    def __init__(self, cfg: SpeechConfig) -> None:
        self.cfg = cfg
        self._q: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._interrupt = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._proc: subprocess.Popen | None = None
        self._backend = "print" if cfg.backend == "print" else _pick_backend()
        self.engine = {"winrt": "Windows OneCore TTS", "sapi": "Windows SAPI TTS", "say": "macOS say", "print": "text only"}[
            self._backend
        ]
        threading.Thread(target=self._worker, daemon=True, name="tts").start()

    # public API --------------------------------------------------------
    def say(self, text: str, language: str | None = None) -> None:
        text = clean_for_speech(text)
        if not text:
            return
        self._interrupt.clear()
        self._idle.clear()
        self._q.put((text, language or detect_language(text)))

    def stop(self) -> None:
        self._interrupt.set()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._kill_current()

    def wait(self) -> None:
        self._q.join()

    @property
    def speaking(self) -> bool:
        return not self._idle.is_set()

    # internals ---------------------------------------------------------
    def _worker(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item and not self._interrupt.is_set():
                    self._speak(*item)
            except Exception:
                log.exception("TTS failed")
            finally:
                self._q.task_done()
                if self._q.empty():
                    self._idle.set()

    def _speak(self, text: str, lang: str) -> None:
        print(f"  🔊 {text}", flush=True)
        if self._backend == "say":
            voice = (self.cfg.voice_hi or "Lekha") if lang == "hi" else self.cfg.voice_en
            cmd = ["say", "-r", str(int(190 * self.cfg.rate))]
            if voice:
                cmd += ["-v", voice]
            self._run([*cmd, text])
        elif self._backend == "winrt":
            wav = asyncio.run(_winrt_synthesize(text, lang, self.cfg))
            _play_wav_bytes(wav, self._interrupt)
        elif self._backend == "sapi":
            rate = max(-10, min(10, round((self.cfg.rate - 1) * 10)))
            safe = text.replace("'", "''")
            ps = (
                "Add-Type -AssemblyName System.Speech;"
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                f"$s.Rate={rate};$s.Speak('{safe}')"
            )
            self._run(["powershell", "-NoProfile", "-Command", ps])

    def _run(self, cmd: list[str]) -> None:
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._proc.wait()
        self._proc = None

    def _kill_current(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass


def _pick_backend() -> str:
    if sys.platform == "darwin":
        return "say"
    if sys.platform == "win32":
        try:
            import winrt.windows.media.speechsynthesis  # noqa: F401

            return "winrt"
        except ImportError:
            return "sapi"
    return "print"


async def _winrt_synthesize(text: str, lang: str, cfg: SpeechConfig) -> bytes:
    from winrt.windows.media.speechsynthesis import SpeechSynthesizer
    from winrt.windows.storage.streams import DataReader

    synth = SpeechSynthesizer()
    wanted = cfg.voice_hi if lang == "hi" else cfg.voice_en
    for v in SpeechSynthesizer.all_voices:
        if (wanted and wanted.lower() in v.display_name.lower()) or (
            not wanted and v.language.lower().startswith("hi" if lang == "hi" else "en-in")
        ):
            synth.voice = v
            break
    synth.options.speaking_rate = max(0.5, min(6.0, cfg.rate))
    stream = await synth.synthesize_text_to_stream_async(text)
    reader = DataReader(stream.get_input_stream_at(0))
    size = stream.size
    await reader.load_async(size)
    data = bytearray(size)
    reader.read_bytes(data)
    return bytes(data)


def _play_wav_bytes(wav: bytes, interrupt: threading.Event) -> None:
    import io
    import wave

    import numpy as np
    import sounddevice as sd

    with wave.open(io.BytesIO(wav)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch)
    sd.play(pcm, sr)
    duration = len(pcm) / sr
    if interrupt.wait(timeout=duration + 0.1):
        sd.stop()
