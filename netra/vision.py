"""What Netra can see: the screen (with the mouse pointer marked), the webcam, and
OS-native OCR for verbatim reading (Windows.Media.Ocr on Windows, Apple Vision on macOS)."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
from dataclasses import dataclass

from PIL import Image, ImageDraw

log = logging.getLogger("netra")


@dataclass
class Screen:
    image: Image.Image
    pointer: tuple[int, int] | None  # pointer position in image pixels


def capture_screen(mark_pointer: bool = True) -> Screen:
    """Grab the monitor that contains the mouse pointer.

    Set NETRA_SCREEN_IMAGE=path.png to substitute a fixed image (tests, demo recordings).
    """
    if fake := os.environ.get("NETRA_SCREEN_IMAGE"):
        return Screen(Image.open(fake).convert("RGB"), None)
    import mss

    with mss.mss() as sct:  # mss makes the process DPI-aware on Windows, so do this first
        px, py = _pointer_position()
        mons = sct.monitors[1:] or sct.monitors
        mon = next(
            (m for m in mons if m["left"] <= px < m["left"] + m["width"] and m["top"] <= py < m["top"] + m["height"]),
            mons[0],
        )
        shot = sct.grab(mon)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    scale = img.width / mon["width"]  # Retina / HiDPI: image pixels per logical unit
    pointer = (round((px - mon["left"]) * scale), round((py - mon["top"]) * scale))
    if not (0 <= pointer[0] < img.width and 0 <= pointer[1] < img.height):
        pointer = None
    if mark_pointer and pointer:
        r = max(18, img.width // 90)
        d = ImageDraw.Draw(img)
        for w in range(max(3, r // 5)):
            d.ellipse((pointer[0] - r - w, pointer[1] - r - w, pointer[0] + r + w, pointer[1] + r + w), outline=(255, 0, 0))
    return Screen(img, pointer)


def crop_around(screen: Screen, width: int = 900, height: int = 560) -> Image.Image:
    """Zoomed view around the pointer, for "what is under my mouse?"."""
    if not screen.pointer:
        return screen.image
    x, y = screen.pointer
    img = screen.image
    left = min(max(0, x - width // 2), max(0, img.width - width))
    top = min(max(0, y - height // 2), max(0, img.height - height))
    return img.crop((left, top, left + width, top + height))


def _pointer_position() -> tuple[int, int]:
    try:
        from pynput.mouse import Controller

        x, y = Controller().position
        return int(x), int(y)
    except Exception:  # no display server / permission: fall back to the primary monitor
        return 0, 0


def capture_camera(index: int = 0, warmup_frames: int = 8) -> Image.Image:
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("camera support needs the 'camera' extra: uv sync --extra camera") from e
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"camera {index} could not be opened")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame = None
        for _ in range(warmup_frames):  # let auto-exposure settle
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            raise RuntimeError("camera returned no frames")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


# ---------------------------------------------------------------- OCR


@dataclass
class OcrLine:
    text: str
    x: float
    y: float  # top edge, image pixels


def ocr_engine_name() -> str:
    return {"win32": "Windows.Media.Ocr", "darwin": "Apple Vision"}.get(sys.platform, "none")


def ocr(img: Image.Image, language: str = "en") -> list[OcrLine]:
    """Text lines in reading order. Returns [] if no OCR engine is available."""
    try:
        if sys.platform == "win32":
            lines = asyncio.run(_ocr_windows(img, language))
        elif sys.platform == "darwin":
            lines = _ocr_macos(img, language)
        else:
            return []
    except Exception as e:
        log.warning("OCR failed: %s", e)
        return []
    return reading_order(lines)


def reading_order(lines: list[OcrLine], line_tol: float = 12.0) -> list[OcrLine]:
    lines = sorted(lines, key=lambda l: l.y)
    rows: list[list[OcrLine]] = []
    for line in lines:
        if rows and abs(rows[-1][0].y - line.y) <= line_tol:
            rows[-1].append(line)
        else:
            rows.append([line])
    return [l for row in rows for l in sorted(row, key=lambda l: l.x)]


def ocr_text(img: Image.Image, language: str = "en", limit: int = 3000) -> str:
    text = "\n".join(l.text for l in ocr(img, language))
    return text[:limit]


def _ocr_macos(img: Image.Image, language: str) -> list[OcrLine]:
    from ocrmac import ocrmac

    prefs = ["hi-IN", "en-US"] if language == "hi" else ["en-US"]
    results = ocrmac.OCR(img, recognition_level="accurate", language_preference=prefs).recognize()
    out = []
    for text, _conf, (x, y, _w, h) in results:  # normalised, origin bottom-left
        out.append(OcrLine(text, x * img.width, (1 - y - h) * img.height))
    return out


async def _ocr_windows(img: Image.Image, language: str) -> list[OcrLine]:
    from winrt.windows.globalization import Language
    from winrt.windows.graphics.imaging import BitmapDecoder
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    stream = InMemoryRandomAccessStream()
    writer = DataWriter(stream)
    writer.write_bytes(buf.getvalue())
    await writer.store_async()
    writer.detach_stream()
    stream.seek(0)
    bitmap = await (await BitmapDecoder.create_async(stream)).get_software_bitmap_async()

    engine = None
    tag = {"hi": "hi-IN", "en": "en-US"}.get(language)
    if tag and OcrEngine.is_language_supported(Language(tag)):
        engine = OcrEngine.try_create_from_language(Language(tag))
    engine = engine or OcrEngine.try_create_from_user_profile_languages()
    if engine is None:
        raise RuntimeError("no Windows OCR language pack installed")
    result = await engine.recognize_async(bitmap)
    out = []
    for line in result.lines:
        rects = [w.bounding_rect for w in line.words]
        out.append(OcrLine(line.text, min(r.x for r in rects), min(r.y for r in rects)))
    return out
