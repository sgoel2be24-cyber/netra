"""The pipeline: listen -> transcribe (NPU) -> route -> see (screen/camera + OCR)
-> reason (VLM on NPU, streamed) -> speak sentence by sentence."""

from __future__ import annotations

import io
import logging
import threading
import time

from PIL import Image

from netra import audio, vision
from netra.asr import ASR, load_asr
from netra.brain import Brain, BrainUnavailable
from netra.config import Config
from netra.events import bus
from netra.router import HELP_TEXT, Intent, route
from netra.speech import SentenceBuffer, Speaker, detect_language

log = logging.getLogger("netra")

READ_CHUNK_LINES = 12  # verbatim reading is spoken in chunks so "stop" stays responsive


class Assistant:
    def __init__(self, cfg: Config, load_speech_model: bool = True) -> None:
        self.cfg = cfg
        self.brain = Brain(cfg.brain)
        self.speaker = Speaker(cfg.speech)
        self.asr: ASR | None = load_asr(cfg.asr) if load_speech_model else None
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._recording_stop: threading.Event | None = None
        self.last_answer = ""
        self.history: list[dict] = []
        self.engines = {
            "asr": self.asr.engine if self.asr else "off",
            "brain": self.brain.engine,
            "ocr": vision.ocr_engine_name(),
            "tts": self.speaker.engine,
        }
        bus.emit("engines", **self.engines)

    # entry points --------------------------------------------------------
    def talk(self) -> None:
        """Hotkey handler: start listening, or stop an in-progress recording."""
        if self._recording_stop is not None:
            self._recording_stop.set()
            return
        self.interrupt()
        threading.Thread(target=self._listen_and_handle, daemon=True).start()

    def run_command(self, intent: Intent, text: str = "", language: str = "en") -> None:
        self.interrupt()
        threading.Thread(target=self._handle, args=(text, language, intent), daemon=True).start()

    def ask_text(self, text: str) -> None:
        """Typed input (dashboard / CLI); same pipeline as voice minus ASR."""
        self.interrupt()
        threading.Thread(target=self._handle, args=(text, detect_language(text)), daemon=True).start()

    def interrupt(self) -> None:
        self._cancel.set()
        self.speaker.stop()

    # pipeline ------------------------------------------------------------
    def _listen_and_handle(self) -> None:
        if self.asr is None:
            return
        with self._lock:
            self._cancel.clear()
            self._recording_stop = threading.Event()
            bus.emit("status", state="listening")
            audio.earcon("listen")
            time.sleep(0.2)  # don't record our own earcon
            try:
                pcm = audio.record_until_silence(
                    self.cfg.asr.silence_s, self.cfg.asr.max_record_s, stop_event=self._recording_stop
                )
            finally:
                self._recording_stop = None
            if pcm.size == 0:
                bus.emit("status", state="idle")
                self.speaker.say("I didn't hear anything.", "en")
                return
            audio.earcon("think")
            with bus.stage("asr", self.asr.engine):
                tr = self.asr.transcribe(pcm)
        text = tr.text.strip()
        bus.emit("transcript", text=text, language=tr.language)
        if not text:
            self.speaker.say("Sorry, I couldn't make that out.", "en")
            bus.emit("status", state="idle")
            return
        self._handle(text, "hi" if tr.language == "hi" or detect_language(text) == "hi" else "en")

    def _handle(self, text: str, language: str, intent: Intent | None = None) -> None:
        with self._lock:
            self._cancel.clear()
            intent = intent or route(text)
            bus.emit("intent", intent=intent.value, text=text, language=language)
            log.info("intent=%s lang=%s text=%r", intent.value, language, text)
            try:
                self._dispatch(intent, text, language)
            except BrainUnavailable as e:
                log.error("%s", e)
                audio.earcon("error")
                bus.emit("error", message=str(e))
                self.speaker.say("My language model is not running. Please start the model server.", "en")
            except Exception as e:
                log.exception("request failed")
                audio.earcon("error")
                bus.emit("error", message=str(e))
                self.speaker.say(f"Sorry, something went wrong: {e}", "en")
            finally:
                bus.emit("status", state="idle")

    def _dispatch(self, intent: Intent, text: str, lang: str) -> None:
        sp = self.speaker
        if intent is Intent.STOP:
            return  # interrupt() already silenced speech
        if intent is Intent.REPEAT:
            sp.say(self.last_answer or "There is nothing to repeat yet.", lang)
            return
        if intent is Intent.HELP:
            sp.say(HELP_TEXT.get(lang, HELP_TEXT["en"]), lang)
            return
        if intent in (Intent.SLOWER, Intent.FASTER):
            step = 0.2 if intent is Intent.FASTER else -0.2
            self.cfg.speech.rate = round(min(2.0, max(0.6, self.cfg.speech.rate + step)), 1)
            sp.say(f"Speech rate {self.cfg.speech.rate}.", "en")
            return

        if intent is Intent.CAMERA:
            with bus.stage("camera", "webcam"):
                img = vision.capture_camera(self.cfg.camera_index)
            self._publish_image(img)
            self._reason(text or "What is in front of me?", lang, img, camera=True)
            return

        with bus.stage("capture", "screen"):
            screen = vision.capture_screen(mark_pointer=intent is not Intent.READ_SCREEN)
        self._publish_image(screen.image)

        if intent is Intent.READ_SCREEN:
            with bus.stage("ocr", vision.ocr_engine_name()):
                lines = [l.text for l in vision.ocr(screen.image, lang)]
            if not lines:  # no OCR engine or nothing found: let the VLM read it
                self._reason("Read all the text on this screen, top to bottom.", lang, screen.image)
                return
            self.last_answer = "\n".join(lines)
            bus.emit("answer", text=self.last_answer, done=True)
            for i in range(0, len(lines), READ_CHUNK_LINES):
                if self._cancel.is_set():
                    break
                sp.say(". ".join(lines[i : i + READ_CHUNK_LINES]))
            return

        if intent is Intent.POINTER:
            view = vision.crop_around(screen)
            self._publish_image(view)
            self._reason(text or "What is under my mouse pointer?", lang, view, pointer_marked=True)
            return

        with bus.stage("ocr", vision.ocr_engine_name()):
            ocr = vision.ocr_text(screen.image, lang)
        context = f"Text found on screen by OCR, in reading order (may contain errors):\n{ocr}" if ocr else ""
        question = text if intent is Intent.ASK_SCREEN else "Describe this screen for me."
        self._reason(question, lang, screen.image, context=context, pointer_marked=screen.pointer is not None)

    def _reason(
        self,
        question: str,
        lang: str,
        image: Image.Image | None,
        context: str = "",
        camera: bool = False,
        pointer_marked: bool = False,
    ) -> None:
        bus.emit("status", state="thinking")
        sentences = SentenceBuffer()
        parts: list[str] = []
        t0 = time.perf_counter()
        first = True
        with bus.stage("vlm", self.brain.engine):
            for delta in self.brain.ask(question, lang, image, context, camera, self.history[-4:], pointer_marked):
                if self._cancel.is_set():
                    break
                parts.append(delta)
                bus.emit("answer", text="".join(parts), done=False)
                for s in sentences.feed(delta):
                    if first:
                        bus.emit("metric", name="time_to_first_speech_ms", value=round((time.perf_counter() - t0) * 1000))
                        bus.emit("status", state="speaking")
                        first = False
                    self.speaker.say(s, lang)
        if not self._cancel.is_set():
            for s in sentences.flush():
                self.speaker.say(s, lang)
        answer = "".join(parts).strip()
        self.last_answer = answer
        bus.emit("answer", text=answer, done=True)
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]

    def _publish_image(self, img: Image.Image) -> None:
        buf = io.BytesIO()
        thumb = img.convert("RGB")
        thumb.thumbnail((960, 960))
        thumb.save(buf, format="JPEG", quality=80)
        bus.emit("image", jpeg=buf.getvalue())
