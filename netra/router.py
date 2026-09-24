"""Map an utterance (English, Hindi or Hinglish) to an intent with fast, predictable rules.

Rules beat an LLM classifier here: they add zero latency, never hallucinate a command,
and a blind user can learn them. Anything unrecognised is treated as a question about
the current screen, which is what most requests are.
"""

from __future__ import annotations

import re
from enum import StrEnum


class Intent(StrEnum):
    STOP = "stop"
    REPEAT = "repeat"
    HELP = "help"
    SLOWER = "slower"
    FASTER = "faster"
    READ_SCREEN = "read_screen"
    DESCRIBE_SCREEN = "describe_screen"
    POINTER = "pointer"
    CAMERA = "camera"
    ASK_SCREEN = "ask_screen"


def _words(*phrases: str) -> re.Pattern[str]:
    # `\b` is unreliable around Devanagari vowel signs, so only guard Latin letters.
    return re.compile(r"(?<![a-z])(?:" + "|".join(phrases) + r")(?![a-z])")


# Control commands only fire on short utterances, so "which laptop is faster?" is a question.
_CONTROL = [
    (Intent.STOP, _words("stop", "quiet", "enough", "cancel", "shut up", "bas", "ruko", "chup", "रुको", "बस", "चुप")),
    (Intent.REPEAT, _words("repeat", "again", "phir se", "dobara", "दोबारा", "फिर से")),
    (Intent.HELP, _words("help", "what can you do", "commands", "madad", "मदद")),
    (Intent.SLOWER, _words("slower", "slow down", "dheere", "धीरे")),
    (Intent.FASTER, _words("faster", "speed up", "tez", "jaldi", "तेज़", "तेज", "जल्दी")),
]

_TASKS = [
    (
        Intent.CAMERA,
        _words(
            "camera", "webcam", "in front of me", "am i holding", "in my hand",
            r"this (?:note|currency|medicine|tablet|strip|packet|bottle|label|letter|paper|bill|envelope|book)",
            r"(?:which|what) (?:note|currency|medicine|tablet)",
            "kaun sa note", "saamne", "samne", "haath", "कैमरा", "सामने", "हाथ", "नोट",
        ),
    ),
    (Intent.POINTER, _words("mouse", "cursor", "pointer", "माउस", "कर्सर")),
    (
        Intent.READ_SCREEN,
        _words(
            r"read (?:out |aloud )?(?:the |this |my |all )?(?:screen|page|text|window|everything)",
            "read aloud", "read it all", "padho", "padh kar", "पढ़ो", "पढ़कर", "पढ़ कर",
        ),
    ),
    (
        Intent.DESCRIBE_SCREEN,
        _words(
            "describe", r"what(?:'s| is) on (?:my |the )?screen", "what do you see", "where am i",
            "screen par kya", "kya dikh", "स्क्रीन पर क्या", "क्या दिख",
        ),
    ),
]


def route(text: str) -> Intent:
    t = text.lower().strip(" .!?।")
    if len(t.split()) <= 5:
        for intent, pattern in _CONTROL:
            if pattern.search(t):
                return intent
    for intent, pattern in _TASKS:
        if pattern.search(t):
            return intent
    return Intent.ASK_SCREEN


HELP_TEXT = {
    "en": (
        "Press Control Alt Space and ask me anything about your screen. "
        "Say describe the screen, read the screen, or what is under my mouse. "
        "Say camera, or what am I holding, to use the webcam. "
        "Say stop, repeat, slower or faster at any time. "
        "Shortcuts: Control Alt D describes, R reads, C uses the camera, A repeats, S stops."
    ),
    "hi": (
        "कंट्रोल ऑल्ट स्पेस दबाइए और स्क्रीन के बारे में कुछ भी पूछिए। "
        "कहिए स्क्रीन पर क्या है, या स्क्रीन पढ़ो। "
        "कैमरा के लिए कहिए, मेरे हाथ में क्या है। "
        "रुको, दोबारा, धीरे या तेज़ कभी भी कह सकते हैं।"
    ),
}
