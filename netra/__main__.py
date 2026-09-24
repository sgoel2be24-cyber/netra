"""Netra command line.

  netra run                     global hotkeys + dashboard (the real app)
  netra ask "question" [--image F | --screen | --camera] [--mute]
  netra transcribe F.wav        speech-to-text only
  netra doctor                  check NPU, model server, OCR, TTS
  netra bench                   latency of each stage on this machine
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import statistics
import sys
import threading
import time
from pathlib import Path

from netra import __version__, device
from netra.config import ROOT, Config


def main(argv: list[str] | None = None) -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")  # Hindi text in the console
    p = argparse.ArgumentParser(prog="netra", description="Offline voice + vision assistant on the Snapdragon NPU")
    p.add_argument("--config", help="path to netra.toml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="start hotkeys and the dashboard")
    r.add_argument("--no-hotkeys", action="store_true", help="dashboard only (no global keyboard hook)")

    a = sub.add_parser("ask", help="one-shot question")
    a.add_argument("question", nargs="?", default="", help="typed question (or use --audio)")
    a.add_argument("--audio", help="WAV file with a spoken question (runs speech-to-text first)")
    src = a.add_mutually_exclusive_group()
    src.add_argument("--image", help="image file to ask about")
    src.add_argument("--screen", action="store_true", help="use a live screenshot")
    src.add_argument("--camera", action="store_true", help="use the webcam")
    a.add_argument("--no-ocr", action="store_true", help="skip OCR grounding")
    a.add_argument("--lang", default=None, help="en or hi (default: detect from the question)")
    a.add_argument("--mute", action="store_true", help="print only, don't speak")

    t = sub.add_parser("transcribe", help="speech-to-text on a WAV file")
    t.add_argument("wav")

    sub.add_parser("doctor", help="check hardware and services")

    b = sub.add_parser("bench", help="measure per-stage latency")
    b.add_argument("--runs", type=int, default=3)
    b.add_argument("--wav", default=str(ROOT / "samples" / "question.wav"))
    b.add_argument("--image", default=str(ROOT / "samples" / "error_dialog.png"))
    b.add_argument("--out", help="write results as JSON")

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "huggingface_hub", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = Config.load(args.config)
    {"run": cmd_run, "ask": cmd_ask, "transcribe": cmd_transcribe, "doctor": cmd_doctor, "bench": cmd_bench}[
        args.cmd
    ](cfg, args)


def cmd_run(cfg: Config, args) -> None:
    from pynput import keyboard

    from netra.assistant import Assistant
    from netra.dashboard import serve
    from netra.router import Intent

    ok, why = device.npu_status()
    logging.info("Snapdragon NPU: %s (%s)", "yes" if ok else "no", why)
    netra = Assistant(cfg)
    serve(netra, cfg.dashboard_port)
    hk = cfg.hotkeys
    bindings = {
        hk.talk: netra.talk,
        hk.describe_screen: lambda: netra.run_command(Intent.DESCRIBE_SCREEN),
        hk.read_screen: lambda: netra.run_command(Intent.READ_SCREEN),
        hk.camera: lambda: netra.run_command(Intent.CAMERA),
        hk.repeat: lambda: netra.run_command(Intent.REPEAT),
        hk.stop: netra.interrupt,
    }
    labels = ["talk (press again to stop)", "describe screen", "read screen", "camera", "repeat", "stop speaking"]
    print(f"\nNetra {__version__} is running. Dashboard: http://127.0.0.1:{cfg.dashboard_port}")
    for combo, label in zip(bindings, labels, strict=True):
        print(f"  {combo:<22} {label}")
    netra.speaker.say("Netra is ready. Press Control Alt Space to talk.", "en")
    try:
        if args.no_hotkeys:
            threading.Event().wait()
        with keyboard.GlobalHotKeys(bindings) as listener:
            listener.join()
    except KeyboardInterrupt:
        pass


def cmd_ask(cfg: Config, args) -> None:
    from PIL import Image

    from netra import vision
    from netra.brain import Brain
    from netra.speech import SentenceBuffer, Speaker, detect_language

    question = args.question
    if args.audio:
        from netra.asr import load_asr
        from netra.audio import load_wav

        asr = load_asr(cfg.asr)
        t0 = time.perf_counter()
        tr = asr.transcribe(load_wav(args.audio))
        print(f"[{asr.engine}: {(time.perf_counter() - t0) * 1000:.0f} ms] heard ({tr.language}): {tr.text}")
        question = tr.text
    if not question:
        sys.exit("ask: give a question or --audio FILE")
    lang = args.lang or detect_language(question)
    img = None
    camera = False
    if args.image:
        img = Image.open(args.image)
    elif args.screen:
        img = vision.capture_screen().image
    elif args.camera:
        img, camera = vision.capture_camera(cfg.camera_index), True
    context = ""
    if img is not None and not args.no_ocr and not camera:
        t0 = time.perf_counter()
        text = vision.ocr_text(img, lang)
        print(f"[ocr {vision.ocr_engine_name()}: {len(text)} chars, {(time.perf_counter() - t0) * 1000:.0f} ms]")
        if text:
            context = f"Text found on screen by OCR, in reading order (may contain errors):\n{text}"
    brain = Brain(cfg.brain)
    speaker = None if args.mute else Speaker(cfg.speech)
    buf = SentenceBuffer()
    t0 = time.perf_counter()
    first = None
    for delta in brain.ask(question, lang, img, context, camera):
        first = first or time.perf_counter()
        print(delta, end="", flush=True)
        if speaker:
            for s in buf.feed(delta):
                speaker.say(s, lang)
    total = time.perf_counter() - t0
    print(f"\n[{brain.engine}: first token {((first or t0) - t0) * 1000:.0f} ms, total {total * 1000:.0f} ms]")
    if speaker:
        for s in buf.flush():
            speaker.say(s, lang)
        speaker.wait()


def cmd_transcribe(cfg: Config, args) -> None:
    from netra.asr import load_asr
    from netra.audio import load_wav

    asr = load_asr(cfg.asr)
    pcm = load_wav(args.wav)
    t0 = time.perf_counter()
    tr = asr.transcribe(pcm)
    print(f"[{tr.language}] {tr.text}")
    print(f"[{asr.engine}: {(time.perf_counter() - t0) * 1000:.0f} ms for {len(pcm) / 16000:.1f} s of audio]")


def cmd_doctor(cfg: Config, args) -> None:
    from netra import vision
    from netra.brain import Brain

    def row(ok: bool | None, name: str, detail: str) -> None:
        mark = {True: "OK  ", False: "FAIL", None: "INFO"}[ok]
        print(f"  [{mark}] {name:<14} {detail}")

    print(f"Netra {__version__} doctor")
    row(None, "machine", f"{device.cpu_name()} · {platform.system()} {platform.machine()} · Python {platform.python_version()}")
    row(device.is_snapdragon() or None, "snapdragon", "yes" if device.is_snapdragon() else "no (dev fallbacks will be used)")
    if device.python_is_emulated():
        row(False, "python", "x64 Python under emulation cannot use the NPU; install ARM64 Python")
    ok, why = device.npu_status()
    row(ok if device.is_snapdragon() else None, "npu (QNN EP)", why)
    model_dir = Path(cfg.asr.model_dir)
    has = (model_dir / "encoder.onnx").is_file() and (model_dir / "decoder.onnx").is_file()
    row(has if device.is_snapdragon() else None, "whisper npu", f"{model_dir} {'found' if has else 'missing'}")
    b_ok, b_why = Brain(cfg.brain).health()
    row(b_ok, "brain (VLM)", b_why)
    row(vision.ocr_engine_name() != "none", "ocr", vision.ocr_engine_name())
    try:
        import sounddevice as sd

        row(True, "audio", f"in: {sd.query_devices(kind='input')['name']} · out: {sd.query_devices(kind='output')['name']}")
    except Exception as e:
        row(False, "audio", str(e))
    try:
        import cv2  # noqa: F401

        row(True, "camera", "opencv available")
    except ImportError:
        row(None, "camera", "not installed (uv sync --extra camera)")


def cmd_bench(cfg: Config, args) -> None:
    from PIL import Image

    from netra import vision
    from netra.asr import load_asr
    from netra.audio import load_wav
    from netra.brain import Brain

    results: dict = {"machine": device.cpu_name(), "snapdragon": device.is_snapdragon(), "stages": {}}

    def timeit(name: str, engine: str, fn, runs: int) -> None:
        fn()  # warm-up (graph load / cache)
        ms = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            ms.append((time.perf_counter() - t0) * 1000)
        results["stages"][name] = {"engine": engine, "median_ms": round(statistics.median(ms)), "runs": runs}
        print(f"  {name:<18} {statistics.median(ms):8.0f} ms   {engine}")

    print(f"Benchmark on {results['machine']}")
    if Path(args.wav).is_file():
        asr = load_asr(cfg.asr)
        pcm = load_wav(args.wav)
        timeit("asr", asr.engine, lambda: asr.transcribe(pcm), args.runs)
    img = Image.open(args.image)
    timeit("ocr", vision.ocr_engine_name(), lambda: vision.ocr_text(img), args.runs)
    brain = Brain(cfg.brain)
    first: list[float] = []

    def vlm() -> None:
        # Nudge one pixel per run so the server's prompt/image cache can't skip the real work.
        frame = img.convert("RGB")
        frame.putpixel((random.randrange(frame.width), random.randrange(frame.height)), (random.randrange(256),) * 3)
        t0 = time.perf_counter()
        got = False
        for _ in brain.ask("Describe this screen for me.", "en", frame):
            if not got:
                first.append((time.perf_counter() - t0) * 1000)
                got = True

    timeit("vlm_total", brain.engine, vlm, args.runs)
    results["stages"]["vlm_first_token"] = {"engine": brain.engine, "median_ms": round(statistics.median(first[1:] or first))}
    print(f"  {'vlm_first_token':<18} {statistics.median(first[1:] or first):8.0f} ms   {brain.engine}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
