"""
src/language_detector.py
=========================
Lightweight language detection for Silvie — German vs English.

Strategy (no external libraries needed):
  1. German-specific Unicode characters (ä ö ü ß) → fast hard signal
  2. Keyword overlap scoring — German vs English common words
  3. Defaults to German ("de") when ambiguous (primary market is DACH)

Returns
-------
  "de"  — German
  "en"  — English
"""

import re
import logging
from typing import Literal

logger = logging.getLogger(__name__)

Language = Literal["de", "en"]

# ── German indicator words (not shared with English) ────────────────────────
_DE_WORDS = frozenset({
    "ich", "sie", "wir", "ihr", "und", "oder", "aber", "der", "die",
    "das", "ein", "eine", "ist", "bin", "bist", "sind", "war", "mit",
    "für", "auf", "bei", "von", "aus", "nach", "wie", "was", "wer",
    "bitte", "danke", "hallo", "tschüss", "gut", "sehr", "auch", "nicht",
    "können", "haben", "werden", "möchte", "meine", "ihre", "ja", "nein",
    "wo", "wann", "warum", "welche", "alle", "viel", "mehr", "noch",
    "schon", "immer", "nie", "hier", "da", "dort", "dann", "jetzt",
    "heute", "morgen", "wieder", "etwas", "alles", "nichts", "jemand",
    "über", "unter", "zwischen", "durch", "ohne", "gegen",
})

# ── English indicator words (not shared with German) ────────────────────────
_EN_WORDS = frozenset({
    "i", "you", "he", "she", "they", "we", "the", "a", "an", "and",
    "or", "but", "is", "are", "am", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "shall",
    "please", "thank", "thanks", "hello", "bye", "goodbye",
    "good", "great", "yes", "no", "not", "very", "more", "some",
    "what", "who", "where", "when", "why", "how", "which",
    "my", "your", "his", "her", "our", "their", "its",
    "this", "that", "these", "those", "with", "for", "from",
    "about", "need", "want", "help", "tell", "show", "find",
    "also", "just", "now", "here", "there", "then", "still",
    "all", "any", "each", "every", "both", "few", "many",
    "how", "fine", "doing", "going", "today", "great", "well",
    "sure", "right", "got", "get", "let", "know",
    "think", "feel", "look",
})

# ── German-specific characters that are a strong signal ─────────────────────
_DE_CHARS = frozenset("äöüÄÖÜß")


def detect_language(text: str) -> Language:
    if not text or not text.strip():
        return "de"

    de_char_count = sum(1 for ch in text if ch in _DE_CHARS)
    if de_char_count >= 1:
        return "de"

    tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    de_score = len(tokens & _DE_WORDS)
    en_score = len(tokens & _EN_WORDS)

    if en_score > de_score:
        return "en"

    return "de"


# ── Simple demo ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("Hallo Silvie, wie geht es dir?",                  "de"),
        ("Hello Silvie, what is youwel?",                   "en"),
        ("Ich möchte nach Wien reisen.",                    "de"),
        ("I would like to travel to Vienna.",               "en"),
        ("Was ist youwel genau?",                           "de"),
        ("Can you help me find a doctor?",                  "en"),
        ("Meine Daten — sind die sicher?",                  "de"),
        ("Is my personal data safe with youwel?",          "en"),
        ("Tell me about Silvie.",                           "en"),
        ("Ich brauche einen Steuerberater.",                "de"),
    ]
    for text, expected in tests:
        detected = detect_language(text)
        status   = "✅" if detected == expected else "❌"
        print(f"{status} [{detected}] {text[:55]}")