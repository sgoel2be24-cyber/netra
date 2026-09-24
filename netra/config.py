"""Settings, loaded from ``netra.toml`` (repo root or ``~/.netra.toml``) with sane defaults.

Every field can be overridden with an env var: ``NETRA_<SECTION>_<KEY>``,
e.g. ``NETRA_BRAIN_MODEL=google/gemma-4-E2B-it-qat-q4_0-gguf``.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from netra import device

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("NETRA_CACHE", Path.home() / ".cache" / "netra"))


@dataclass
class BrainConfig:
    # OpenAI-compatible endpoint. On Snapdragon this is `geniex serve` (Hexagon NPU);
    # on a dev Mac it is llama.cpp's `llama-server` (Metal).
    url: str = ""
    model: str = ""
    max_tokens: int = 320
    temperature: float = 0.2
    image_max_side: int = 1280
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if not self.url:
            self.url = "http://127.0.0.1:18181/v1" if device.is_snapdragon() else "http://127.0.0.1:8080/v1"
        if not self.model:
            # GenieX-verified on the X Elite NPU: image + audio, strong Hindi.
            self.model = (
                "google/gemma-4-E2B-it-qat-q4_0-gguf" if device.is_snapdragon() else "qwen3-vl-2b"
            )


@dataclass
class AsrConfig:
    # "auto" picks Whisper on the NPU (QNN EP) when available, else faster-whisper on CPU.
    backend: str = "auto"
    model_size: str = "base"  # whisper size: tiny/base/small
    model_dir: str = str(CACHE / "whisper_npu")
    language: str = "auto"  # auto | en | hi
    silence_s: float = 1.1  # stop recording after this much trailing silence
    max_record_s: float = 15.0


@dataclass
class SpeechConfig:
    backend: str = "auto"  # auto | print (text only, for tests)
    rate: float = 1.0  # 0.5 .. 2.0 (blind users often prefer fast speech)
    voice_en: str = ""  # substring match on voice name; "" = system default
    voice_hi: str = ""


@dataclass
class HotkeyConfig:
    talk: str = "<ctrl>+<alt>+<space>"
    describe_screen: str = "<ctrl>+<alt>+d"
    read_screen: str = "<ctrl>+<alt>+r"
    camera: str = "<ctrl>+<alt>+c"
    repeat: str = "<ctrl>+<alt>+a"
    stop: str = "<ctrl>+<alt>+s"


@dataclass
class Config:
    brain: BrainConfig = field(default_factory=BrainConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    dashboard_port: int = 8765
    camera_index: int = 0

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        data: dict = {}
        for p in [path, ROOT / "netra.toml", Path.home() / ".netra.toml"]:
            if p and Path(p).is_file():
                data = tomllib.loads(Path(p).read_text(encoding="utf-8"))
                break
        cfg = cls(
            brain=_build(BrainConfig, data.get("brain", {}), "BRAIN"),
            asr=_build(AsrConfig, data.get("asr", {}), "ASR"),
            speech=_build(SpeechConfig, data.get("speech", {}), "SPEECH"),
            hotkeys=_build(HotkeyConfig, data.get("hotkeys", {}), "HOTKEYS"),
        )
        cfg.dashboard_port = int(os.environ.get("NETRA_DASHBOARD_PORT", data.get("dashboard_port", 8765)))
        cfg.camera_index = int(os.environ.get("NETRA_CAMERA_INDEX", data.get("camera_index", 0)))
        return cfg


def _build(kind: type, values: dict, section: str):
    kwargs = {}
    for f in fields(kind):
        raw = os.environ.get(f"NETRA_{section}_{f.name.upper()}", values.get(f.name))
        if raw is None:
            continue
        default_type = type(f.default) if f.default is not None else str
        kwargs[f.name] = default_type(raw) if default_type in (int, float) else raw
    return kind(**kwargs)
