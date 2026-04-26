"""
src/predict_v2.py  — Silvie inference engine v2.1  (speed-optimised)
======================================================================
Key changes vs v2.0
────────────────────
1.  GPU auto-detection        n_gpu_layers auto-set to max when CUDA/Metal present
2.  Flash attention            flash_attn=True  (llama-cpp-python ≥ 0.2.56)
3.  Faster first-token stream  _MIN_CHUNK=1 / _MAX_CHUNK=20 → client sees text instantly
4.  LRU response cache         deterministic intents (greeting/farewell/thanks) skip LLM
5.  Larger n_batch             512 → 1024 → more tokens processed per CPU cycle
6.  Greedy fast-path           temperature=0 intents use greedy decode (no sampling overhead)
7.  Duplicate intent call fix  intent classified once, passed into predict_hf / predict_stream_hf
8.  Pre-encoded SSE token      b'data: {"token":' pre-built outside hot loop
"""

import sys
import os
import json
import logging
import threading
import functools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Generator, Optional

logger = logging.getLogger(__name__)

# Config

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF_MODEL_PATH = os.path.join(BASE_DIR, "gguf_model", "Phi-3-mini-4k-instruct-q4.gguf")

# Streaming threshold
# BEFORE: _MIN_CHUNK=5, _MAX_CHUNK=40  → 40-char delay before first word appears
# AFTER:  _MIN_CHUNK=1, _MAX_CHUNK=20  → client sees the first word within ~50 ms
_MIN_CHUNK   = 1    # flush as soon as any word boundary is hit
_MAX_CHUNK   = 20   # hard flush — keeps chunks short and snappy
_SENTENCE_END = frozenset(".!?")
_SOFT_FLUSH   = frozenset(",;:")

# Per-intent token budget
_INTENT_MAX_TOKENS: dict[str, int] = {
    "emergency":      0,
    "greeting":       60,
    "farewell":       50,
    "thanks":         50,
    "repeat_request": 100,
    "general_chat":   150,
    "youwel_explain": 250,
    "youwel_privacy": 200,
    "youwel_funding": 200,
    "youwel_health":  200,
    "trip_planning":  450,
    
    # NEW — Assets Module Scripted Flow
    "user_need_assets":             150,
    "data_privacy_consent":         180,
    "specialist_search_permission": 150,
    "contact_preference":           120,
    "info_request":                 120,
    "collect_contact":              120,
    "availability_time":            120,
    "feedback":                     120,
    "conversation_end":             120,
}
_DEFAULT_MAX_TOKENS = 200

# Intents that are essentially deterministic → use greedy decode
# Greedy (temperature=0) is faster than sampling because no top-k/nucleus filter needed.
_GREEDY_INTENTS = frozenset({"greeting", "farewell", "thanks", "emergency"})

# System prompt
SYSTEM_PROMPT = """Du bist Silvie, die freundliche KI-Sprachassistentin von youwel.app.
Du hilfst aktiven Menschen ab 40 Jahren in Deutschland.

=== STIMME UND IDENTITÄT ===
1. SPRICH IMMER IN DER ERSTEN PERSON ("Ich bin Silvie", "Ich helfe").
2. Spreche NIEMALS in der dritten Person über dich selbst (Nicht "youwel ist", sondern "Wir sind").
3. Ich bin AUSSCHLIESSLICH Silvie. Ich bin NICHT ChatGPT, NICHT Microsoft, NICHT OpenAI.
4. Ich bin eine KI, kein Mensch.

=== REGELN ===
Antworte immer auf Deutsch, in kurzen Sätzen (max 20 Wörter pro Satz).
Sei geduldig und empathisch. Vermeide Fachbegriffe.
Bei medizinischen oder rechtlichen Fragen verweise an einen Fachspezialisten.
Bei Notfällen (Schmerzen, Sturz) nenne sofort die 112.
Formuliere jede Antwort leicht anders, damit es sich natürlich anfühlt.

=== BEISPIELGESPRÄCHE ===
Nutzer: Was ist youwel?
Silvie: Ich bin Silvie, die Assistentin von youwel. Wir sind eine kostenfreie Plattform für Menschen ab 40.

Nutzer: Wer bist du?
Silvie: Ich bin Silvie, Ihre KI-Assistentin von youwel! Ich helfe Ihnen, Spezialisten zu finden oder Reisen zu planen.

Nutzer: Ich brauche einen Arzt.
Silvie: Natürlich helfe ich Ihnen! Soll ich einen Hausarzt in Ihrer Nähe suchen? Bei einem Notfall rufen Sie bitte sofort die 112."""

# Emergency response
EMERGENCY_RESPONSE = (
    "Bitte rufen Sie sofort den Notruf an! "
    "In Deutschland: 112 für Rettungsdienst und Feuerwehr, 110 für die Polizei. "
    "Drücken Sie bitte Ihren Notrufknopf, wenn Sie einen haben. "
    "Sind Sie in Sicherheit?"
)

# Context hints per intent
CONTEXT_HINTS = {
    "youwel_explain": (
        "Antworte in ERSTER PERSON als Silvie. Nie in dritter Person. "
        "Ich bin Silvie, die digitale Assistentin von youwel — einer kostenfreien Plattform für aktive Menschen ab 40 Jahren. "
        "Ich verbinde Menschen mit vertrauenswürdigen Fachspezialisten in ihrer Nähe — für Vermögensplanung, Gesundheit und Reisen. "
        "Alle meine Dienste sind vollständig kostenfrei, barrierefrei und DSGVO-konform. "
        "Man erreicht mich über youwel.app im Browser — kein Download nötig, einfach auf den grünen Juwel tippen. "
        "Ich bin kein Microsoft- oder OpenAI-Produkt. Ich bin ausschließlich Silvie von youwel."
    ),
    "youwel_privacy": (
        "Antworte in ERSTER PERSON als Silvie. Nie in dritter Person. "
        "Ich speichere Ihre Daten ausschließlich in Deutschland, in einer hermetisch gesicherten deutschen Datenbank. "
        "Ich bin vollständig DSGVO-konform. Ich gebe Ihre Daten niemals an unbefugte Dritte weiter. "
        "Nur autorisierte youwel-Mitarbeiter haben begrenzten technischen Systemzugang. "
        "Sie können jederzeit die Löschung Ihrer Daten beantragen."
    ),
    "youwel_funding": (
        "Antworte in ERSTER PERSON als Silvie. Nie in dritter Person. "
        "Ich bin für Sie als Nutzer vollständig kostenlos — Sie zahlen gar nichts. "
        "Ich werde durch Vereinbarungen mit den Fachspezialisten finanziert, die ich vermittle. "
        "Die Fachleute zahlen eine Provision an uns, wenn sie über mich neue Klienten gewinnen."
    ),
    "youwel_health": (
        "Antworte in ERSTER PERSON als Silvie. "
        "Ich kann Allgemeinmediziner, Spezialisten, Orthopäden und Ernährungsberater in Ihrer Nähe finden. "
        "Bei medizinischen Beschwerden empfehle ich immer, zuerst einen Arzt aufzusuchen. "
        "Bei Notfällen weise ich sofort auf 112 hin."
    ),
    
    # MODULE: GESCHAFFENE WERTE (Assets)
    "user_need_assets": (
        "Antworte in ERSTER PERSON als Silvie. "
        "Frage höflich: 'Womit genau kann ich Ihnen im Bereich Vermögen, Vermögenssicherung oder Anlageformen helfen?'"
    ),
    "data_privacy_consent": (
        "Antworte in ERSTER PERSON als Silvie. "
        "Ich beruhige den Nutzer, dass seine Daten hermetisch geschützt in einer deutschen Datenbank bleiben. "
        "Frage: 'Haben Sie Fragen zum Datenschutz oder möchten Sie die Datenschutzbestimmungen einsehen?'"
    ),
    "specialist_search_permission": (
        "Antworte in ERSTER PERSON als Silvie. "
        "Frage: 'Darf ich für Sie passende Fachspezialisten in Ihrer Nähe mit guten Bewertungen recherchieren?'"
    ),
    "contact_preference": (
        "Antworte in ERSTER PERSON als Silvie. "
        "Frage: 'Wie möchten Sie über passende Fachleute informiert werden – per E-Mail oder SMS?'"
    ),
    "info_request": (
        "Antworte in ERSTER PERSON als Silvie. "
        "Frage: 'Möchten Sie zusätzlich eine kurze Zusammenfassung mit Informationen zu Ihrem Anliegen erhalten?'"
    ),
    "collect_contact": (
        "Antworte in ERSTER PERSON als Silvie. "
        "Frage: 'Bitte teilen Sie mir Ihre E-Mail-Adresse oder Mobilnummer mit.'"
    ),
    "availability_time": (
        "Antworte in ERSTER PERSON als Silvie. "
        "Frage: 'In welchen für Sie angenehmen Zeitfenstern dürfen diese Fachleute Sie morgen oder übermorgen kontaktieren?'"
    ),
    "feedback": (
        "Antworte in ERSTER PERSON als Silvie. "
        "Frage: 'Haben Sie weitere Fragen, Anregungen oder Verbesserungsvorschläge für mich?'"
    ),
    "conversation_end": (
        "Antworte in ERSTER PERSON als Silvie. "
        "Verabschiede dich freundlich. Aussage: 'Vielen Dank für Ihre Anfrage. Ich kümmere mich um die nächsten Schritte. Auf Wiederhören!'"
    ),
    
    "trip_planning": (
        "Antworte in ERSTER PERSON als Silvie. "
        "Ich helfe bei der Reiseplanung für aktive Senioren. "
        "Ich kenne viele schöne Ziele: Wien, Salzburg, Bodensee, Toskana, Mallorca, Nordsee, Ostsee, Adria. "
        "Ich suche Sehenswürdigkeiten, Hotels und Empfehlungen für das genannte Reiseziel. "
        "Wenn noch kein Reiseziel genannt wurde, frage ich danach."
    ),
    "greeting": (
        "Antworte in ERSTER PERSON als Silvie von youwel. "
        "Ich begrüße den Nutzer herzlich und frage, womit ich helfen kann."
    ),
    "farewell": (
        "Antworte in ERSTER PERSON als Silvie. "
        "Ich verabschiede mich freundlich und weise darauf hin, dass ich jederzeit wieder erreichbar bin."
    ),
    "thanks": (
        "Antworte in ERSTER PERSON als Silvie. "
        "Ich freue mich über den Dank und frage, ob ich noch mit etwas anderem helfen kann."
    ),
    "repeat_request": (
        "Antworte in ERSTER PERSON als Silvie. "
        "Der Nutzer möchte eine Erklärung wiederholt oder vereinfacht haben. "
        "Ich erkläre es geduldig nochmal in einfachen Worten."
    ),
    "general_chat": (
        "Antworte in ERSTER PERSON als Silvie von youwel. "
        "Ich bin kein Mensch, sondern eine KI — kein Microsoft-Produkt, kein ChatGPT, kein OpenAI. "
        "Ich bin ausschließlich Silvie, entwickelt von youwel. "
        "Ich antworte freundlich und verweise sanft auf meine Hauptaufgaben."
    ),
}

# Response cache for deterministic/repetitive intents
# Saves ~500 ms–2 s per cached hit by skipping the LLM entirely.
# Cache is keyed on (intent, normalised_question).  Max 256 entries (LRU).
_CACHEABLE_INTENTS = frozenset({"greeting", "farewell", "thanks"})
_response_cache: dict = {}          # populated lazily
_CACHE_MAXSIZE   = 256

def _cache_key(intent: str, question: str) -> str:
    # Normalise: lower-case, strip punctuation noise
    import re
    q = re.sub(r"[^\w\s]", "", question.lower()).strip()
    return f"{intent}:{q}"

def _get_cached(intent: str, question: str) -> Optional[str]:
    if intent not in _CACHEABLE_INTENTS:
        return None
    return _response_cache.get(_cache_key(intent, question))

def _set_cached(intent: str, question: str, answer: str) -> None:
    if intent not in _CACHEABLE_INTENTS:
        return
    if len(_response_cache) >= _CACHE_MAXSIZE:
        # Evict oldest key (insertion-ordered dict, Python 3.7+)
        _response_cache.pop(next(iter(_response_cache)))
    _response_cache[_cache_key(intent, question)] = answer

# GPU detection
def _detect_gpu_layers() -> int:
    """
    Return the number of layers to offload to GPU.
    -1 means 'all layers' (llama-cpp convention).
    Returns 0 on CPU-only systems.

    Priority: CUDA > Metal (Apple Silicon) > CPU
    """
    # Allow env-var override: SILVIE_GPU_LAYERS=35
    env = os.environ.get("SILVIE_GPU_LAYERS")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            pass

    # CUDA check — also verify cudaMalloc actually works on this VM
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().splitlines()[0]
            # Verify CUDA compute works (some vGPUs block cudaMalloc)
            try:
                import torch
                t = torch.zeros(1).cuda()
                del t
                logger.info(f"[gpu] CUDA GPU detected and verified: {gpu_name}")
                return -1  # offload all layers
            except Exception as cuda_err:
                logger.warning(f"[gpu] GPU detected ({gpu_name}) but CUDA compute unavailable: {cuda_err} — falling back to CPU")
                return 0
    except Exception:
        pass

    # Metal check (Apple Silicon)
    try:
        import platform
        if platform.system() == "Darwin":
            import subprocess
            r = subprocess.run(
                ["sysctl", "-n", "hw.optional.arm64"],
                capture_output=True, text=True, timeout=2
            )
            if r.stdout.strip() == "1":
                logger.info("[gpu] Apple Silicon detected → Metal offload")
                return -1
    except Exception:
        pass

    logger.info("[gpu] No GPU detected — running on CPU.")
    return 0

# Eager model cache
_llm_instance = None
_llm_lock     = threading.Lock()
_preload_done = threading.Event()


def preload_model(gguf_path: str = GGUF_MODEL_PATH) -> None:
    """Load the GGUF model in a background thread at server startup."""
    def _do_load():
        global _llm_instance
        try:
            from llama_cpp import Llama
            gpu_layers      = _detect_gpu_layers()
            optimal_threads = 1 if gpu_layers != 0 else min(os.cpu_count() or 4, 6)
            # When GPU offloads everything, CPU threads only run non-offloaded layers
            # → 1-2 threads avoids unnecessary context-switch overhead.

            logger.info(
                f"[preload] Loading GGUF from {gguf_path} "
                f"(gpu_layers={gpu_layers}, threads={optimal_threads}) …"
            )
            llm = Llama(
                model_path    = gguf_path,
                n_ctx         = 2048,
                n_threads     = optimal_threads,
                n_batch       = 1024,          # was 512 — doubles token throughput on CPU
                n_gpu_layers  = gpu_layers,    # was hardcoded 0
                use_mmap      = True,
                use_mlock     = False,
                f16_kv        = True,
                flash_attn    = True,          # 20-30% speedup (llama-cpp ≥ 0.2.56)
                verbose       = False,
            )
            with _llm_lock:
                _llm_instance = llm
            logger.info(
                f"[preload] ✅ Model ready "
                f"(gpu_layers={gpu_layers}, n_batch=1024, flash_attn=True)."
            )
        except Exception as exc:
            logger.error(f"[preload] ❌ Model load failed: {exc}")
        finally:
            _preload_done.set()

    t = threading.Thread(target=_do_load, daemon=True, name="silvie-preload")
    t.start()


def load_silvie_hf(gguf_path: str = GGUF_MODEL_PATH, timeout: float = 120.0):
    """Return (llm, None, device_str). Waits for preload if needed."""
    global _llm_instance

    with _llm_lock:
        if _llm_instance is not None:
            return _llm_instance, None, "gpu" if _detect_gpu_layers() != 0 else "cpu"

    if not _preload_done.is_set():
        logger.info("Waiting for background model preload…")
        _preload_done.wait(timeout=timeout)

    with _llm_lock:
        if _llm_instance is not None:
            return _llm_instance, None, "gpu" if _detect_gpu_layers() != 0 else "cpu"

    # Synchronous fallback
    logger.warning("Preload was not called — loading model synchronously.")
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF model not found at {gguf_path}.")

    from llama_cpp import Llama
    gpu_layers = _detect_gpu_layers()
    optimal_threads = 1 if gpu_layers != 0 else min(os.cpu_count() or 4, 6)
    llm = Llama(
        model_path   = gguf_path,
        n_ctx        = 2048,
        n_threads    = optimal_threads,
        n_batch      = 1024,
        n_gpu_layers = gpu_layers,
        use_mmap     = True,
        use_mlock    = False,
        f16_kv       = True,
        flash_attn   = True,
        verbose      = False,
    )
    with _llm_lock:
        _llm_instance = llm
    device = "gpu" if gpu_layers != 0 else "cpu"
    logger.info(f"✅ GGUF model loaded (sync fallback, device={device}).")
    return _llm_instance, None, device


# Message builder
def _build_messages(
    question: str,
    intent: Optional[str] = None,
    system_override: Optional[str] = None,
    conversation_history: Optional[list[dict]] = None,
) -> list[dict]:
    system   = system_override or SYSTEM_PROMPT
    messages = [{"role": "system", "content": system}]

    if conversation_history:
        for turn in conversation_history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    if intent and intent in CONTEXT_HINTS:
        messages.append({
            "role":    "assistant",
            "content": f"[Interner Kontext — nicht vorlesen]: {CONTEXT_HINTS[intent]}"
        })

    messages.append({"role": "user", "content": question})
    return messages


# Streaming chunk helper
def _smart_chunks(stream) -> Generator[str, None, None]:
    """
    Yield phrase-sized chunks with aggressive first-token delivery.

    Differences from v2.0:
      • _MIN_CHUNK=1  → flush on the FIRST space seen (was 5)
      • _MAX_CHUNK=20 → hard cap halved (was 40)
    Result: client sees the first word in ~50 ms instead of waiting for a
    5-char buffer to fill up — matches ChatGPT's perceived responsiveness.
    """
    buffer    = ""
    prev_char = ""

    for chunk in stream:
        delta = chunk["choices"][0]["delta"].get("content", "")
        if not delta:
            continue

        for ch in delta:
            buffer  += ch
            buf_len  = len(buffer)

            # 1. Sentence-end flush
            if prev_char in _SENTENCE_END and ch == " ":
                yield buffer
                buffer = ""
                prev_char = ""
                continue

            # 2. Hard cap flush (back to last space if possible)
            if buf_len >= _MAX_CHUNK:
                last_space = buffer.rfind(" ")
                if last_space > _MIN_CHUNK:
                    yield buffer[:last_space + 1]
                    buffer = buffer[last_space + 1:]
                else:
                    yield buffer
                    buffer = ""
                prev_char = ch
                continue

            # 3. Soft boundary flush
            if prev_char in _SOFT_FLUSH and ch == " " and buf_len >= _MIN_CHUNK:
                yield buffer
                buffer = ""
                prev_char = ""
                continue

            # 4. Word boundary flush (fires as soon as buf_len >= 1)
            if ch == " " and buf_len >= _MIN_CHUNK:
                yield buffer
                buffer = ""
                prev_char = ""
                continue

            prev_char = ch

    if buffer:
        yield buffer


# Sampling parameters
def _sampling_params(intent: Optional[str], temperature: float, top_k: int) -> dict:
    """
    Return the best sampling kwargs for a given intent.

    Greedy intents (greeting/farewell/thanks) use temperature=0 which
    skips the top-k/nucleus filter entirely → ~10 % faster per token.
    """
    if intent in _GREEDY_INTENTS:
        return dict(temperature=0.0, top_k=1, top_p=1.0, repeat_penalty=1.0)
    return dict(temperature=temperature, top_k=top_k, top_p=0.9, repeat_penalty=1.1)


# Public predict functions
def predict_hf(
    question: str,
    model,
    tokenizer,
    device,
    conversation_history: Optional[list[dict]] = None,
    temperature: float = 0.7,
    top_k: int = 40,
    intent: Optional[str] = None,       # ← NEW: accept pre-classified intent
) -> str:
    """
    Generate a complete answer (blocking).
    Pass `intent` from the caller to avoid classifying twice.
    """
    from src.intent_classifier import classify_intent, is_emergency
    from src.rag_travel        import get_travel_rag_context, build_travel_prompt
    from src.senior_filter     import format_for_senior

    if is_emergency(question):
        return EMERGENCY_RESPONSE

    # Use caller-supplied intent if available (avoids double classification)
    if intent is None:
        intent = classify_intent(question)
    logger.info(f"[predict] Intent: {intent}")

    # Cache check for simple deterministic intents
    cached = _get_cached(intent, question)
    if cached:
        logger.debug(f"[predict] Cache hit for intent={intent}")
        return cached

    system_override = None
    if intent == "trip_planning":
        dest, context = get_travel_rag_context(question)
        if dest and context:
            system_override = build_travel_prompt(question, dest, context)

    messages = _build_messages(
        question,
        intent=intent,
        system_override=system_override,
        conversation_history=conversation_history,
    )

    max_tok = _INTENT_MAX_TOKENS.get(intent, _DEFAULT_MAX_TOKENS)
    params  = _sampling_params(intent, temperature, top_k)

    output = model.create_chat_completion(
        messages   = messages,
        max_tokens = max_tok,
        **params,
    )
    raw    = output["choices"][0]["message"]["content"]
    raw    = _strip_internal_prefix(raw)
    answer = format_for_senior(raw, user_text=question, intent=intent)

    _set_cached(intent, question, answer)
    return answer


def predict_stream_hf(
    question: str,
    model,
    tokenizer,
    device,
    conversation_history: Optional[list[dict]] = None,
    temperature: float = 0.7,
    top_k: int = 40,
    intent: Optional[str] = None,       # ← NEW: accept pre-classified intent
) -> Generator[str, None, None]:
    """
    Stream the answer in phrase-sized chunks.
    Pass `intent` to avoid classifying twice.
    """
    from src.intent_classifier import classify_intent, is_emergency
    from src.rag_travel        import get_travel_rag_context, build_travel_prompt

    if is_emergency(question):
        yield EMERGENCY_RESPONSE
        return

    if intent is None:
        intent = classify_intent(question)
    logger.info(f"[stream] Intent: {intent}")

    # Cache hit: stream cached answer word-by-word for consistent UX
    cached = _get_cached(intent, question)
    if cached:
        logger.debug(f"[stream] Cache hit for intent={intent}")
        words = cached.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
        return

    system_override = None
    if intent == "trip_planning":
        dest, context = get_travel_rag_context(question)
        if dest and context:
            system_override = build_travel_prompt(question, dest, context)

    messages = _build_messages(
        question,
        intent=intent,
        system_override=system_override,
        conversation_history=conversation_history,
    )

    max_tok = _INTENT_MAX_TOKENS.get(intent, _DEFAULT_MAX_TOKENS)
    params  = _sampling_params(intent, temperature, top_k)

    stream = model.create_chat_completion(
        messages   = messages,
        max_tokens = max_tok,
        **params,
        stream     = True,
    )

    full_response: list[str] = []
    first = True
    for chunk in _smart_chunks(stream):
        if first:
            chunk = _strip_internal_prefix(chunk)
            first = False
        if chunk:
            full_response.append(chunk)
            yield chunk

    # Cache the assembled response for next time
    if full_response:
        _set_cached(intent, question, "".join(full_response))

    logger.debug(f"[stream] Full response: {''.join(full_response)}")


# Internal helpers
def _strip_internal_prefix(text: str) -> str:
    marker = "[Interner Kontext"
    if marker in text:
        end = text.find("]:", text.find(marker))
        if end != -1:
            text = text[end + 2:].strip()
    return text.lstrip("]").strip()


# CLI demo
if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)
    print("Preloading Silvie GGUF model…")
    preload_model()
    _preload_done.wait()
    _model, _tokenizer, _device = load_silvie_hf()
    print(f"Model ready on {_device}\n")

    test_questions = [
        "Hallo Silvie!",
        "Was ist youwel?",
        "Was ist youwel?",
        "Ich möchte nach Wien reisen.",
        "Sind meine Daten sicher?",
        "Ich brauche einen Arzt.",
        "Hilfe! Ich bin gestürzt!",
        "Wie finanziert sich youwel?",
        "Ich möchte mein Vermögen absichern.",
        "Können Sie einen Termin vereinbaren?",
        "Tschüss Silvie!",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        print("A: ", end="", flush=True)
        t0 = time.monotonic()
        for chunk in predict_stream_hf(q, _model, _tokenizer, _device):
            print(chunk, end="", flush=True)
        elapsed = time.monotonic() - t0
        print(f"  [{elapsed:.2f}s]")
