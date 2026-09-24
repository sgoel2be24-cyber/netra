import base64
import io

import pytest
from PIL import Image

from netra.brain import encode_image
from netra.config import Config
from netra.router import Intent, route
from netra.speech import SentenceBuffer, clean_for_speech, detect_language
from netra.vision import OcrLine, reading_order


@pytest.mark.parametrize(
    "text, intent",
    [
        ("Stop", Intent.STOP),
        ("रुको", Intent.STOP),
        ("say that again", Intent.REPEAT),
        ("दोबारा", Intent.REPEAT),
        ("speak faster", Intent.FASTER),
        ("Which of these two laptops is faster and cheaper overall?", Intent.ASK_SCREEN),
        ("Read the screen", Intent.READ_SCREEN),
        ("स्क्रीन पढ़ो", Intent.READ_SCREEN),
        ("Describe my screen", Intent.DESCRIBE_SCREEN),
        ("स्क्रीन पर क्या है?", Intent.DESCRIBE_SCREEN),
        ("What is under my mouse?", Intent.POINTER),
        ("Which note is this?", Intent.CAMERA),
        ("मेरे हाथ में क्या है?", Intent.CAMERA),
        ("What does this error say?", Intent.ASK_SCREEN),
        ("Where is the submit button?", Intent.ASK_SCREEN),
    ],
)
def test_route(text, intent):
    assert route(text) is intent


def test_sentence_buffer_streams_complete_sentences():
    buf = SentenceBuffer()
    out = []
    for delta in ["Your upload fa", "iled. The file is 4.8", " MB. बकाया राशि ₹ 2,340 है।", " Try again"]:
        out += buf.feed(delta)
    assert out == ["Your upload failed.", "The file is 4.8 MB.", "बकाया राशि ₹ 2,340 है।"]
    assert buf.flush() == ["Try again"]


def test_speech_cleanup_and_language():
    assert clean_for_speech("**Error:** see https://x.y/z") == "Error: see a link"
    assert detect_language("अंतिम तिथि") == "hi"
    assert detect_language("deadline") == "en"


def test_reading_order_groups_rows_left_to_right():
    lines = [OcrLine("Save", 1300, 874), OcrLine("Title", 100, 10), OcrLine("Back", 1100, 876)]
    assert [l.text for l in reading_order(lines)] == ["Title", "Back", "Save"]


def test_encode_image_downscales():
    url = encode_image(Image.new("RGB", (3000, 1500)), max_side=1000)
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert img.size == (1000, 500)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("NETRA_BRAIN_MAX_TOKENS", "99")
    monkeypatch.setenv("NETRA_SPEECH_RATE", "1.4")
    cfg = Config.load("/nonexistent.toml")
    assert cfg.brain.max_tokens == 99 and cfg.speech.rate == 1.4


def test_endpointing_stops_after_trailing_silence(monkeypatch):
    import numpy as np
    import sounddevice as sd

    from netra import audio

    sr = audio.SAMPLE_RATE
    rng = np.random.default_rng(0)
    # 0.5 s room noise, 1.5 s "speech", then 5 s of noise the recorder should never reach
    signal = np.concatenate([
        0.002 * rng.standard_normal(sr // 2),
        0.2 * np.sin(np.linspace(0, 2 * np.pi * 220 * 1.5, int(sr * 1.5))),
        0.002 * rng.standard_normal(sr * 5),
    ]).astype(np.float32)
    clock = {"t": 0.0}

    class FakeStream:
        def __init__(self, **kw):
            self.pos = 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            block = signal[self.pos : self.pos + n]
            self.pos += n
            clock["t"] += n / sr
            return block.reshape(-1, 1), False

    monkeypatch.setattr(sd, "InputStream", FakeStream)
    monkeypatch.setattr(audio.time, "monotonic", lambda: clock["t"])
    pcm = audio.record_until_silence(silence_s=1.0, max_s=15)
    assert 2.8 < len(pcm) / sr < 3.3  # 0.5 noise + 1.5 speech + ~1.0 silence
