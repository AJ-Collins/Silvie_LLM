"""
Silvie inference engine - no intent, unlimited responses, persistent memory
"""

from __future__ import annotations

import logging
import os
import threading
import re
from typing import Generator, List, Optional, Tuple

import torch
from transformers import TextIteratorStreamer
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)

_HERE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CKPT_PATH = os.path.join(_HERE, "model", "llama-checkpoint")

_model_instance     = None
_tokenizer_instance = None
_preload_done       = False
_load_lock          = threading.Lock()
_response_cache: dict[str, str] = {}
_CACHE_MAX = 512

# ── System prompt — no restrictions, full answers ────────────────────────────
_SILVIE_SYSTEM = {
    "de": """Du bist Silvie, die freundliche KI-Assistentin von youwel.app.
Du hilfst aktiven Menschen ab 60 Jahren in Deutschland und Europa.
Die Zielgruppe von youwel sind Menschen ab 60 Jahren — nicht 87, nicht 70, sondern ab 60.
Beantworte ALLE Fragen vollständig, hilfreich und mit konkreten Details.
Gib immer praktische Beispiele und nützliche Tipps.
Erinnere dich an den bisherigen Gesprächsverlauf und antworte im Kontext.""",

    "en": """You are Silvie, a friendly AI assistant from youwel.app.
You help active people aged 60 and above across Germany and Europe.
youwel's target audience is people aged 60+ — always say 60+, never any other age.
Answer ALL questions fully, helpfully, and with concrete details.
Always provide practical examples and actionable advice.
Remember the full conversation and refer back to earlier messages naturally."""
}

MAX_NEW_TOKENS  = 600   # single limit, no intent needed
MAX_HISTORY_TURNS = 10  # how many past turns to include


def _load_model():
    global _model_instance, _tokenizer_instance
    if _model_instance is not None:
        return _model_instance, _tokenizer_instance

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[silvie] Loading checkpoint from {_CKPT_PATH} on {device}…")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = _CKPT_PATH,
        max_seq_length = 2048,   # increased — longer history needs more context
        dtype          = None,
        load_in_4bit   = True,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    _model_instance     = model
    _tokenizer_instance = tokenizer
    logger.info("[silvie] ✅ Model ready")
    return model, tokenizer


def preload_model() -> None:
    global _preload_done
    with _load_lock:
        _load_model()
    _preload_done = True


def load_silvie_hf() -> Tuple[object, object, str]:
    model, tokenizer = _load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model, tokenizer, device


def _build_messages(question: str, history: List[dict], language: str) -> List[dict]:
    # 1. Start with the system instruction
    messages = [{"role": "system", "content": _SILVIE_SYSTEM[language]}]
    
    # 2. Append the previous history (these are the dictionaries sent from React)
    for entry in history:
        messages.append({"role": entry["role"], "content": entry["content"]})
    
    # 3. Append the new question
    messages.append({"role": "user", "content": question})
    
    return messages


def _clean(text: str) -> str:
    for tok in ["<|eot_id|>", "<|end_of_text|>", "<|assistant|>"]:
        text = text.replace(tok, "")
    return text.strip()


# ── Blocking predict ──────────────────────────────────────────────────────────
def predict_hf(
    question:             str,
    model:                object,
    tokenizer:            object,
    device:               str,
    conversation_history: List[dict] = None,
    language:             str        = "en",
    # intent kept as kwarg so existing callers don't break — just ignored
    intent:               str        = None,
) -> str:
    history  = conversation_history or []
    messages = _build_messages(question, history, language)

    # Only cache if there's no history (stateless one-shot questions)
    use_cache = len(history) == 0
    cache_key = f"{language}|{question.strip()}"
    if use_cache and cache_key in _response_cache:
        return _response_cache[cache_key]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors        = "pt",
        add_generation_prompt = True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            temperature    = 0.7,
            top_p          = 0.9,
            do_sample      = True,
            use_cache      = True,
            pad_token_id   = tokenizer.pad_token_id,
        )

    answer = tokenizer.decode(
        outputs[0][inputs.shape[1]:], skip_special_tokens=True
    )
    answer = _clean(answer)

    if not answer:
        answer = (
            "Entschuldigung, ich habe das nicht verstanden. Bitte wiederholen Sie das."
            if language == "de"
            else "Sorry, I didn't catch that. Could you rephrase?"
        )

    if use_cache and len(_response_cache) < _CACHE_MAX:
        _response_cache[cache_key] = answer

    return answer


# ── Streaming predict ─────────────────────────────────────────────────────────
def predict_stream_hf(
    question:             str,
    model:                object,
    tokenizer:            object,
    device:               str,
    conversation_history: List[dict] = None,
    language:             str        = "en",
    intent:               str        = None,   # ignored, kept for compat
) -> Generator[str, None, None]:
    history  = conversation_history or []
    messages = _build_messages(question, history, language)

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors        = "pt",
        add_generation_prompt = True,
    ).to(device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt         = True,
        skip_special_tokens = True,
    )

    thread = threading.Thread(
        target  = model.generate,
        kwargs  = dict(
            input_ids      = inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            temperature    = 0.7,
            top_p          = 0.9,
            do_sample      = True,
            use_cache      = True,
            pad_token_id   = tokenizer.pad_token_id,
            streamer       = streamer,
        ),
        daemon=True,
    )
    thread.start()

    for token in streamer:
        # Strip out stopping tokens, but preserve ALL normal whitespace and newlines
        clean_tok = token.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").replace("<|assistant|>", "")
        if clean_tok:
            yield clean_tok

    thread.join(timeout=60)
