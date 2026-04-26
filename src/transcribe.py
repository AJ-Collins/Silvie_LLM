"""
src/transcribe.py — Speech-to-text using OpenAI Whisper (local, runs on CPU).

Whisper model sizes and trade-offs:
  tiny   ~39M params  fastest, lower accuracy  — good for prototyping
  base   ~74M params  fast, decent accuracy    — recommended default
  small  ~244M params balanced                 — best for production seniors UX
  medium ~769M params high accuracy            — use if GPU available

The model is loaded once on first call and cached for the server lifetime.
"""

import functools
import logging

logger = logging.getLogger(__name__)

# Lazy model cache
_whisper_model = None
WHISPER_MODEL_SIZE = "base"


def _get_model():
    """Load Whisper model once and reuse."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
        except ImportError:
            raise RuntimeError(
                "openai-whisper is not installed. "
                "Run: pip install openai-whisper"
            )
        logger.info(f"Loading Whisper '{WHISPER_MODEL_SIZE}' model…")
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        logger.info("Whisper model loaded.")
    return _whisper_model


# Public API 
def transcribe_audio(file_path: str, language: str = "de") -> str:
    """
    Transcribe an audio file to text using Whisper.

    Parameters
    ----------
    file_path : str
        Path to the audio file (wav / m4a / mp3 / webm).
        React Native's expo-av saves recordings as .m4a by default.
    language : str
        ISO language code hint for Whisper. Defaults to "en".
        Pass None to let Whisper auto-detect (slower).

    Returns
    -------
    str
        Transcribed text, stripped of leading/trailing whitespace.

    Raises
    ------
    RuntimeError
        If Whisper is not installed or transcription fails.
    """
    whisper_model = _get_model()

    options = {
        "fp16": False,          # CPU-safe — no half-precision on CPU
        "language": language,   # skip language detection → faster
        "task": "transcribe",
    }

    try:
        result = whisper_model.transcribe(file_path, **options)
        return result["text"].strip()
    except Exception as exc:
        logger.error(f"Whisper transcription failed for {file_path}: {exc}")
        raise RuntimeError(str(exc)) from exc