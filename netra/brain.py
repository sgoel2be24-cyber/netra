"""The vision-language "brain": a VLM served over an OpenAI-compatible API.

On Snapdragon: `geniex serve` (Qualcomm AI Hub GenieX), model pinned to the Hexagon NPU.
On a dev Mac: llama.cpp `llama-server` running the same GGUF family on Metal.
Only the stdlib is used for HTTP so the ARM64 install stays small and wheel-safe.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from urllib.parse import urlparse

from PIL import Image

from netra import device
from netra.config import BrainConfig

LANG_NAMES = {"en": "English", "hi": "Hindi"}

SYSTEM_PROMPT = """You are Netra, a calm and concise assistant for a blind or low-vision person \
using a computer. Everything you write is read aloud by a text-to-speech voice.

Rules:
- Reply in {lang}. Write plain spoken sentences: no markdown, no bullet symbols, no emojis.
- Lead with the answer. Use at most four short sentences unless the user asks you to read or list \
something in full. Skip details the user did not ask about.
- Describing a screen: say which app or page it is, then the most important content (headings, \
dialog messages, errors), then the key controls (buttons, fields, links) with their position, \
for example "top right" or "bottom left".
- Quote on-screen text exactly when it matters: error messages, prices, dates, names, numbers.
- If something is unreadable or you are unsure, say so instead of guessing. Never say "as you can see".
"""

POINTER_HINT = """A red ring on the screenshot marks the user's mouse pointer. Mention what is under \
it only when it helps the user."""

CAMERA_HINT = """The image comes from the user's webcam. They may be holding up a document, a \
medicine strip, a food packet, or an Indian currency note, or pointing the camera at a room. For \
currency, state the denomination in rupees. For medicines, read the name, strength and expiry \
date. For documents, say what kind of document it is and summarise it. If the object is cut off \
or blurry, tell the user how to move it (closer, left, right, more light)."""


class BrainUnavailable(RuntimeError):
    pass


def encode_image(img: Image.Image, max_side: int) -> str:
    img = img.convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Brain:
    def __init__(self, cfg: BrainConfig) -> None:
        self.cfg = cfg
        host = urlparse(cfg.url).port
        if device.is_snapdragon() and host == 18181:
            self.engine = f"{cfg.model.split('/')[-1]} · Hexagon NPU (GenieX)"
        else:
            self.engine = f"{cfg.model} · {device.cpu_name()} (llama.cpp, dev)"

    def health(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(self.cfg.url.rstrip("/") + "/models", timeout=3) as r:
                ids = [m.get("id") for m in json.load(r).get("data", [])]
                return True, f"{self.cfg.url} serving {', '.join(map(str, ids)) or 'a model'}"
        except (urllib.error.URLError, OSError, ValueError) as e:
            return False, f"{self.cfg.url} unreachable ({e})"

    def ask(
        self,
        question: str,
        language: str = "en",
        image: Image.Image | None = None,
        context: str = "",
        camera: bool = False,
        history: list[dict] | None = None,
        pointer_marked: bool = False,
    ) -> Iterator[str]:
        """Stream the answer as text deltas."""
        system = SYSTEM_PROMPT.format(lang=LANG_NAMES.get(language, "the user's language"))
        if pointer_marked:
            system += "\n" + POINTER_HINT
        if camera:
            system += "\n" + CAMERA_HINT
        user: list[dict] = []
        if image is not None:
            user.append({"type": "image_url", "image_url": {"url": encode_image(image, self.cfg.image_max_side)}})
        text = question
        if context:
            text = f"{context}\n\nUser request: {question}"
        user.append({"type": "text", "text": text})
        messages = [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": user}]
        yield from self._stream(messages)

    def _stream(self, messages: list[dict]) -> Iterator[str]:
        body = json.dumps(
            {
                "model": self.cfg.model,
                "messages": messages,
                "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
                "stream": True,
            }
        ).encode()
        req = urllib.request.Request(
            self.cfg.url.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer netra"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=self.cfg.timeout_s)
        except urllib.error.HTTPError as e:
            raise BrainUnavailable(f"model server error {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, OSError) as e:
            hint = "start it with `geniex serve`" if device.is_snapdragon() else "run scripts/dev_brain.sh"
            raise BrainUnavailable(f"cannot reach the model server at {self.cfg.url}; {hint}") from e
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
