"""Render the demo/benchmark inputs in samples/: a web-form screenshot whose error message is
baked into an image (the kind of content screen readers cannot read), and spoken questions.

Run on macOS (uses `say` for the audio): uv run python scripts/make_samples.py
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "samples"
FONT = "/System/Library/Fonts/Helvetica.ttc"
SYMBOLS = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def font(size: int, bold: bool = False, path: str = FONT) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, index=1 if bold and path == FONT else 0)


def shaped(text: str, size: int, color=(30, 30, 30), bold: bool = False) -> Image.Image:
    """Render Devanagari with CoreText (Pillow wheels lack raqm, so matras would be misplaced)."""
    import AppKit

    name = "KohinoorDevanagari-Semibold" if bold else "KohinoorDevanagari-Regular"
    ns_font = AppKit.NSFont.fontWithName_size_(name, size)
    ns_color = AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(*(c / 255 for c in color), 1.0)
    s = AppKit.NSAttributedString.alloc().initWithString_attributes_(
        text, {AppKit.NSFontAttributeName: ns_font, AppKit.NSForegroundColorAttributeName: ns_color}
    )
    w, h = s.size()
    rep = AppKit.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, int(w) + 4, int(h) + 4, 8, 4, True, False, AppKit.NSDeviceRGBColorSpace, 0, 0
    )
    AppKit.NSGraphicsContext.saveGraphicsState()
    AppKit.NSGraphicsContext.setCurrentContext_(AppKit.NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
    s.drawAtPoint_((2, 2))
    AppKit.NSGraphicsContext.restoreGraphicsState()
    png = rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, None)
    return Image.open(io.BytesIO(bytes(png))).convert("RGBA")


def paste_text(img: Image.Image, xy: tuple[int, int], text: str, size: int, color=(30, 30, 30), bold: bool = False) -> None:
    t = shaped(text, size, color, bold)
    img.paste(t, xy, t)


def error_dialog() -> Image.Image:
    W, H = 1600, 1000
    img = Image.new("RGB", (W, H), (236, 239, 243))
    d = ImageDraw.Draw(img)
    # browser chrome
    d.rectangle((0, 0, W, 86), fill=(222, 225, 230))
    d.rounded_rectangle((180, 22, 1400, 64), 20, fill=(255, 255, 255))
    d.text((210, 31), "https://scholarships.example.gov.in/apply/step-3", font=font(22), fill=(60, 64, 67))
    # site header
    d.rectangle((0, 86, W, 170), fill=(22, 58, 112))
    d.text((60, 108), "National Scholarship Portal  ·  Application 2026-27", font=font(34, True), fill="white")
    # stepper
    for i, (label, done) in enumerate([("1 Profile", True), ("2 Bank details", True), ("3 Documents", False), ("4 Submit", False)]):
        x = 60 + i * 370
        d.rounded_rectangle((x, 200, x + 340, 250), 10, fill=(46, 125, 50) if done else (255, 255, 255), outline=(180, 186, 194))
        d.text((x + 20, 211), label, font=font(24, True), fill="white" if done else (40, 44, 52))
    # form card
    d.rounded_rectangle((60, 280, 1540, 940), 16, fill="white", outline=(210, 214, 220))
    d.text((100, 310), "Upload documents", font=font(36, True), fill=(20, 24, 30))
    rows = [("Income certificate", "income_cert_2026.pdf  (1.2 MB)", True), ("Caste certificate", "caste_certificate_scan.jpg  (4.8 MB)", False), ("Marksheet (Class 12)", "Choose file…", None)]
    for i, (label, value, ok) in enumerate(rows):
        y = 390 + i * 110
        d.text((100, y), label, font=font(26, True), fill=(40, 44, 52))
        d.rounded_rectangle((100, y + 38, 900, y + 88), 8, outline=(160, 166, 174), fill=(250, 250, 251))
        d.text((120, y + 50), value, font=font(24), fill=(60, 64, 67))
        mark, color = {True: ("✓ Uploaded", (46, 125, 50)), False: ("✗ Rejected", (198, 40, 40)), None: ("", (0, 0, 0))}[ok]
        d.text((930, y + 50), mark, font=font(24, path=SYMBOLS), fill=color)
    # the error is an image-only banner: a screen reader gets nothing from it
    d.rounded_rectangle((100, 730, 1500, 820), 12, fill=(253, 236, 234), outline=(198, 40, 40), width=3)
    d.text((130, 746), "Upload failed: Caste certificate must be under 2 MB.", font=font(28, True), fill=(160, 20, 20))
    d.text((130, 784), "Allowed formats: PDF or JPG. Deadline for this step: 31 October 2026.", font=font(22), fill=(120, 30, 30))
    d.rounded_rectangle((1060, 860, 1260, 915), 10, fill=(255, 255, 255), outline=(22, 58, 112), width=2)
    d.text((1115, 873), "Back", font=font(26, True), fill=(22, 58, 112))
    d.rounded_rectangle((1290, 860, 1500, 915), 10, fill=(22, 58, 112))
    d.text((1320, 873), "Save & Next", font=font(26, True), fill="white")
    return img


def hindi_notice() -> Image.Image:
    img = Image.new("RGB", (1400, 700), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 1400, 110), fill=(183, 28, 28))
    paste_text(img, (50, 18), "बिजली बिल भुगतान", 52, (255, 255, 255), bold=True)
    lines = [
        ("उपभोक्ता संख्या: 4410 2287 9931", 40, False),
        ("बकाया राशि: ₹ 2,340", 48, True),
        ("अंतिम तिथि: 5 अक्टूबर 2026", 40, False),
        ("समय पर भुगतान न करने पर ₹ 150 विलंब शुल्क लगेगा।", 34, False),
    ]
    y = 150
    for text, size, bold in lines:
        paste_text(img, (60, y), text, size, bold=bold)
        y += size + 55
    d.rounded_rectangle((1000, 580, 1340, 660), 14, fill=(46, 125, 50))
    paste_text(img, (1045, 590), "अभी भुगतान करें", 34, (255, 255, 255), bold=True)
    return img


def speak(text: str, voice: str, path: Path) -> None:
    subprocess.run(["say", "-v", voice, "-o", str(path), "--data-format=LEI16@16000", text], check=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    error_dialog().save(OUT / "error_dialog.png")
    hindi_notice().save(OUT / "hindi_bill.png")
    if sys.platform == "darwin":
        speak("Why did my upload fail, and what should I do?", "Rishi", OUT / "question.wav")
        speak("इस बिल में कितना पैसा देना है और आखिरी तारीख क्या है?", "Lekha", OUT / "question_hi.wav")
    print("wrote", *sorted(p.name for p in OUT.iterdir()))
