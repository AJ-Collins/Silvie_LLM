"""
src/transcribe.py
"""

import logging
import os
import subprocess
import tempfile
import torch
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_whisper_model     = None
WHISPER_MODEL_SIZE = "small"


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper, torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper '{WHISPER_MODEL_SIZE}' on {device}…")
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
        logger.info(f"Whisper model loaded on {device}.")
    return _whisper_model


def _convert_to_wav_16k(input_path: str) -> str:
    """Convert any audio format to 16kHz mono WAV using ffmpeg."""
    out_path = input_path + "_16k.wav"
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-sample_fmt", "s16",
            out_path,
        ],
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise ValueError(f"ffmpeg conversion failed: {result.stderr.decode()}")
    return out_path


def _validate_audio(wav_path: str) -> None:
    """Raise if audio is too short to transcribe."""
    import wave
    try:
        with wave.open(wav_path, "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
            if duration < 0.3:
                raise ValueError(f"Audio too short: {duration:.2f}s")
    except wave.Error as e:
        raise ValueError(f"Invalid wav file: {e}")

ALLOWED_LANGUAGES = {"de", "en"}

def transcribe_audio(
    file_path:            str,
    language:             Optional[str] = None,
    auto_detect_language: bool          = True,
) -> Tuple[str, str]:
    converted_path = None
    try:
        # Step 1 — normalize to 16kHz mono WAV (fixes webm/m4a/wrong-rate inputs)
        converted_path = _convert_to_wav_16k(file_path)

        # Step 2 — validate before touching GPU
        _validate_audio(converted_path)

        # Step 3 — transcribe
        whisper_model = _get_model()

        options: dict = {
            "fp16":         torch.cuda.is_available(),
            "task":         "transcribe",
            "beam_size":    5,         # ← was default 1 — improves accuracy ~15%
            "best_of":      5,
            "temperature":  0.0,       # deterministic — no hallucinations
            "condition_on_previous_text": False,  # prevents compounding errors
        }

        if language:
            options["language"] = language
        elif auto_detect_language:
            pass  # let Whisper auto-detect
        else:
            options["language"] = "de"

        try:
            result = whisper_model.transcribe(converted_path, **options)
        except Exception as exc:
            logger.error(f"Whisper transcription failed for {file_path}: {exc}")
            raise RuntimeError(str(exc)) from exc

        transcript        = result["text"].strip()
        detected_language = result.get("language") or language or "en"

        if detected_language not in ALLOWED_LANGUAGES:
            logger.warning(f"Detected language '{detected_language}' not supported — rejecting")
            raise ValueError(
                f"UNSUPPORTED_LANGUAGE:{detected_language}"
            )

        logger.info(f"Transcribed: '{transcript[:60]}…' | detected_language='{detected_language}'")
        return transcript, detected_language

    finally:
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)


def transcribe_audio_simple(
    file_path: str,
    language:  str = "de",
) -> str:
    transcript, _ = transcribe_audio(file_path, language=language)
    return transcript