"""
main.py — Silvie AI FastAPI server v5.2  (bilingual DE + EN)
=============================================================
Changes vs v5.1
─────────────────
1.  Language detection        detect_language() called once per request,
                              result passed into predict_hf / predict_stream_hf.
                              Zero extra latency — keyword-based, no ML model.

2.  /voice endpoint           Now uses transcribe_audio() (returns transcript +
                              detected language) instead of transcribe_audio_simple().
                              Detected language forwarded in the response so
                              the client can pass it to the next /predict call.

3.  X-Silvie-Language header  Added to /predict and /predict-stream responses
                              for client-side debug / analytics.

4.  language field in schemas AnswerResponse and TranscriptResponse gain an
                              optional `language` field. Existing clients that
                              don't read it are unaffected.

API endpoint paths and HTTP methods — UNCHANGED.
Response shapes — fully backwards compatible (new fields are Optional).
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import io
import threading
import wave
import array
import torch as _torch
import asyncio
import json
import logging
import tempfile
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Optional
import signal
import re as _re
import unicodedata
from dataclasses import dataclass, field

from fastapi import FastAPI, File, HTTPException, UploadFile, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, Response
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

_server_start = time.monotonic()
_gpu_semaphore = asyncio.Semaphore(1)

# ── Pre-built SSE byte sequences ─────────────────────────────────────────────
_SSE_TOKEN_PREFIX = b'data: {"token":'
_SSE_DONE         = b"data: [DONE]\n\n"


def _sse_token(chunk: str) -> bytes:
    return _SSE_TOKEN_PREFIX + json.dumps(chunk, ensure_ascii=False).encode() + b"}\n\n"


def _sse_json(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


# ── FastAPI lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.predict_v2        import preload_model
    logger.info("🚀 Silvie startup — initiating model preloads…")
    preload_model()
    threading.Thread(
        target=_preload_tts, daemon=True, name="silvie-tts-preload"
    ).start()
    yield
    logger.info("🛑 Silvie shutdown.")


app = FastAPI(
    title       = "Silvie AI API",
    description = (
        "Silvie is the personal AI voice assistant of youwel.app — "
        "built for active people aged 60+ in Germany and Europe.\n\n"
        "**Bilingual:** German (DE) and English (EN) — auto-detected per request.\n"
        "**Model:           Silvie Llama 3.2 3B (QLoRA fine-tuned)"
        "**Streaming:** Phrase-boundary SSE chunks for natural reading flow\n"
        "**youwel covers 9 everyday areas:** Health · Family · Activity · "
        "Everyday challenges · Home · Hobbies · Safety · Created values · Travel"
    ),
    version  = "5.2.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = [
        "https://silvie.youwel.app",
        "https://app.youwel.app",
        "http://localhost:5173",   # local dev
        "http://localhost:3000",   # local dev
    ],
    allow_credentials=True,   # must be False when using wildcard methods/headers
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_model():
    from src.predict_v2 import load_silvie_hf
    return load_silvie_hf()


def _get_language(text: str) -> str:
    try:
        from src.language_detector import detect_language
        return detect_language(text)
    except Exception:
        return "en"   # ← was "de"


# ── Session memory ────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 10
_request_count = 0
_stream_count  = 0

@dataclass
class Session:
    turns: list[dict] = field(default_factory=list)
    last_active: float = field(default_factory=time.monotonic)

_sessions: dict[str, Session] = {}
SESSION_TTL = 4 * 3600

def _cleanup_sessions():
    now = time.monotonic()
    expired = [sid for sid, s in _sessions.items() if now - s.last_active > SESSION_TTL]
    for sid in expired:
        del _sessions[sid]
    if expired:
        logger.info(f"[sessions] Cleaned up {len(expired)} expired sessions")

def get_history(session_id: Optional[str]) -> list[dict]:
    if not session_id or session_id not in _sessions:
        return []
    return _sessions[session_id].turns[-MAX_HISTORY_TURNS:]   # ← .turns not direct index

def save_to_history(session_id: Optional[str], question: str, answer: str) -> None:
    if not session_id:
        return
    if session_id not in _sessions:
        _sessions[session_id] = Session()
    s = _sessions[session_id]
    s.turns.extend([
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ])
    s.last_active = time.monotonic()
    cap = MAX_HISTORY_TURNS * 2 * 2
    if len(s.turns) > cap:
        s.turns = s.turns[-cap:]

# Request / response schemas
class Question(BaseModel):
    text:       str
    session_id: Optional[str] = None
    language:   Optional[str] = None
    conversation_history: Optional[List[dict]] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "Hallo Silvie!",                   "session_id": "user-abc-123"},
                {"text": "Hello Silvie, what is youwel?",   "session_id": "user-xyz-456"},
                {"text": "Ich möchte nach Wien reisen."},
                {"text": "I'd like to plan a trip to Vienna."},
            ]
        }
    }


class AnswerResponse(BaseModel):
    question:   str
    answer:     str
    intent:     Optional[str] = None
    session_id: Optional[str] = None
    language:   Optional[str] = None   # ← NEW (optional — backwards compatible)


class TranscriptResponse(BaseModel):
    transcript: str
    session_id: Optional[str] = None
    language:   Optional[str] = None   # ← NEW detected language from Whisper


# Audio validation
_ALLOWED_AUDIO = (
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/m4a", "audio/mp4", "audio/x-m4a",
    "audio/mpeg", "audio/mp3",
    "audio/webm", "audio/ogg",
    "application/octet-stream",
)


async def _transcribe_upload(audio: UploadFile) -> tuple[str, str]:
    """
    Transcribe uploaded audio.

    Returns
    -------
    (transcript, detected_language)
    """
    ct = (audio.content_type or "").strip()
    if ct and not any(ct.startswith(p) for p in _ALLOWED_AUDIO):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type '{ct}'. Please send wav, m4a, mp3, or webm.",
        )
    ext = os.path.splitext(audio.filename or "rec.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    loop = asyncio.get_event_loop()
    try:
        from src.transcribe import transcribe_audio
        async with _gpu_semaphore: 
            transcript, detected_lang = await loop.run_in_executor(
                None, transcribe_audio, tmp_path
            )
    except Exception as exc:
        err_str = str(exc).lower()
        # CUDA corruption — the context is poisoned, must restart
        if "cuda error" in err_str or "device-side assert" in err_str:
            logger.critical("[transcribe] CUDA context poisoned — resetting TTS instance")
            global _tts_instance
            _tts_instance = None   # ← force TTS to reload fresh on next call
            # Only trigger full restart if reset doesn't help
            threading.Timer(2.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
            raise HTTPException(
                status_code=503,
                detail="Service restarting. Please try again in a few seconds.",
            )

        if err_str.startswith("unsupported_language:"):
            detected = str(exc).split(":", 1)[1].strip()
            # Return a fixed German redirect — client will speak this aloud
            redirect = (
                "Entschuldigung, ich spreche nur Deutsch und Englisch. "
                "Bitte sprechen Sie auf Deutsch oder auf Englisch. "
                "/ Sorry, I only speak German and English. "
                "Please speak in German or English."
            )
            raise HTTPException(
                status_code=422,
                detail={"type": "unsupported_language", "detected": detected, "message": redirect}
            )

        raise HTTPException(status_code=500, detail=f"Transcription error: {exc}")
    finally:
        os.remove(tmp_path)

    if not transcript.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No speech detected. Please speak clearly and try again. / "
                "Keine Sprache erkannt. Bitte sprechen Sie klar und deutlich."
            ),
        )
    return transcript.strip(), detected_lang


# ── POST /predict ─────────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model = AnswerResponse,
    summary        = "Ask Silvie — full blocking answer (DE + EN)",
)
async def ask_silvie(body: Question):
    global _request_count
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    session_id = body.session_id or str(uuid.uuid4())
    history    = get_history(session_id)

    language = body.language or _get_language(body.text)

    from src.predict_v2 import predict_hf

    loop   = asyncio.get_event_loop()
    model, tokenizer, device = _get_model()
    answer = await loop.run_in_executor(
        None,
        lambda: predict_hf(
            question             = body.text,
            model                = model,
            tokenizer            = tokenizer,
            device               = device,
            conversation_history = history,
            intent               = intent,
            language             = language,   # ← NEW
        ),
    )

    save_to_history(session_id, body.text, answer)
    _request_count += 1

    return AnswerResponse(
        question   = body.text,
        answer     = answer,
        session_id = session_id,
        language   = language,    # ← NEW (ignored by old clients)
    )


# ── POST /predict-stream ──────────────────────────────────────────────────────
@app.post(
    "/predict-stream",
    summary     = "Ask Silvie — SSE phrase-stream (DE + EN)",
    description = (
        "Returns `text/event-stream`. Each `data:` line is `{\"token\": \"...\"}` "
        "where *token* is a readable phrase chunk. Stream ends with `data: [DONE]`.\n\n"
        "Language is auto-detected per request. Both German and English are supported.\n\n"
        "**React Native / Expo usage:**\n"
        "```js\n"
        "const es = new EventSource(`${API_URL}/predict-stream`, {\n"
        "  method: 'POST',\n"
        "  headers: { 'Content-Type': 'application/json' },\n"
        "  body: JSON.stringify({ text: userMessage, session_id: sessionId }),\n"
        "});\n"
        "es.addEventListener('message', (e) => {\n"
        "  if (e.data === '[DONE]') { es.close(); return; }\n"
        "  const parsed = JSON.parse(e.data);\n"
        "  if (parsed.token)      setAnswer(prev => prev + parsed.token);\n"
        "  if (parsed.session_id) setSessionId(parsed.session_id);\n"
        "});\n"
        "```"
    ),
)
def ask_silvie_stream(body: Question, background_tasks: BackgroundTasks):
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    session_id = body.session_id or str(uuid.uuid4())
    
    history = body.conversation_history if body.conversation_history else get_history(session_id)

    language = body.language or _get_language(body.text)

    def sse_generator():
        global _stream_count
        from src.predict_v2 import predict_stream_hf
        model, tokenizer, device = _get_model()
        full_answer: list[str] = []
        try:
            for chunk in predict_stream_hf(
                question             = body.text,
                model                = model,
                tokenizer            = tokenizer,
                device               = device,
                conversation_history = history,
                language             = language,
            ):
                if chunk:
                    full_answer.append(chunk)
                    yield _sse_token(chunk)
        except Exception as exc:
            logger.exception("Streaming error")
            yield _sse_json({"error": str(exc)})
        finally:
            if full_answer:
                assembled = "".join(full_answer)
                save_to_history(session_id, body.text, assembled)
                _stream_count += 1
            yield _sse_json({"session_id": session_id})
            yield _SSE_DONE

    return StreamingResponse(
        sse_generator(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Silvie-Language": language,   # ← NEW debug header
        },
    )


# ── POST /voice ───────────────────────────────────────────────────────────────
@app.post(
    "/voice",
    response_model = TranscriptResponse,
    summary        = "Transcribe audio → return transcript + detected language",
    description    = (
        "Transcribes speech to text using Whisper with automatic language detection. "
        "Both German and English speech are supported. "
        "Pass the returned `language` value to your next `/predict` or "
        "`/predict-stream` call for optimum response quality."
    ),
)
async def ask_silvie_voice(
    audio:        UploadFile     = File(...),
    x_session_id: Optional[str] = Header(None),
):
    transcript, detected_lang = await _transcribe_upload(audio)
    session_id                = x_session_id or str(uuid.uuid4())
    return TranscriptResponse(
        transcript = transcript,
        session_id = session_id,
        language   = detected_lang,   # ← NEW — client passes this to /predict
    )

def _silent_wav(duration_samples: int = 4800) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
        wf.writeframes(b'\x00' * duration_samples * 2)
    buf.seek(0)
    return buf.read()

def _floats_to_wav(wav_floats) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
        pcm = array.array("h", [max(-32768, min(32767, int(s * 32767))) for s in wav_floats])
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf.read()

@app.websocket("/ws/tts")
async def tts_websocket(websocket: WebSocket):
    import base64
    import asyncio
    import json

    await websocket.accept()
    logger.info("[WS/TTS] Client connected — persistent connection established")

    # Shared dynamic tracking state scoped strictly to this connection's lifecycle
    current_session_id = 0
    loop = asyncio.get_event_loop()

    # ── HELPERS ──
    def _chunk_long_text(s: str, limit: int) -> list[str]:
        chunks = []
        while len(s) > limit:
            cut = s[:limit].rfind(' ')
            if cut <= 5:
                cut = limit
            chunks.append(s[:cut].strip())
            s = s[cut:].strip()
        if s.strip():
            chunks.append(s.strip())
        return chunks

    def _clean_and_split_text(text: str) -> list[str]:    
        """Consolidated single-pass sanitization and sentence segmentation."""
        import re as _re
        import unicodedata
        
        text = unicodedata.normalize("NFC", text)    
        text = _re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = _re.sub(r'\*(.*?)\*',     r'\1', text)
        text = _re.sub(r'^#{1,6}\s+',    '',    text, flags=_re.MULTILINE)
        text = _re.sub(r'^[-*+]\s+',     '',    text, flags=_re.MULTILINE)
        text = _re.sub(r'^\d+\.\s+(.*?)$', r'\1.', text, flags=_re.MULTILINE)
        text = _re.sub(r'`[^`]*`',       '',    text)
        text = _re.sub(r'[<>\[\]]',      '',    text)
        text = text.replace('\u2019', "'").replace('\u2018', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2014', ',').replace('\u2013', ',')
        text = text.replace('\u2026', '.')
        text = text.replace('&', 'and').replace('%', ' percent')
        text = _re.sub(r'[*_#{}|\\^~`]', '', text)
        text = _re.sub(r'[^\x00-\x7F\u00C0-\u024F]', '', text)
        text = _re.sub(r'\s+', ' ', text).strip()

        MAX_CHUNK = 80        
        parts = _re.split(r'(?<=[.!?])\s+', text)
        result, current = [], ""

        for part in parts:
            if len(part) > MAX_CHUNK:
                if current:
                    result.append(current)
                    current = ""
                subparts = _re.split(r'(?<=[,;])\s+', part)
                for sub in subparts:
                    if len(sub) > MAX_CHUNK:
                        result.extend(_chunk_long_text(sub, MAX_CHUNK))
                    elif sub.strip():
                        result.append(sub.strip())
            elif len(current) + 1 + len(part) <= MAX_CHUNK:
                current = (current + " " + part).strip()
            else:
                if current:
                    result.append(current)
                current = part

        if current:
            result.append(current)

        return [c.strip() for c in result if len(c.strip()) >= 3]

    def _synth_chunk(tts_model, chunk_text: str, voice_wav: str, lang: str) -> bytes:
        # Pre-sanitized text coming through from unified pipeline
        sub_chunks = _chunk_long_text(chunk_text, 75)
        combined_wav_floats = []

        try:
            for sub_chunk in sub_chunks:
                if len(sub_chunk.strip()) < 3:
                    continue
                wav_floats = tts_model.tts(
                    text        = sub_chunk.strip(),
                    speaker_wav = voice_wav,
                    language    = lang,
                )
                if hasattr(wav_floats, "tolist"):
                    combined_wav_floats.extend(wav_floats.tolist())
                else:
                    combined_wav_floats.extend(wav_floats)
        except Exception as exc:
            logger.error(f"[TTS] Generation failure on text '{chunk_text[:30]}': {exc}")
            raise

        if not combined_wav_floats:
            return _silent_wav()

        return _floats_to_wav(combined_wav_floats)

    # ── CONCURRENT CORE PIPELINE ──
    async def process_synthesis(raw_text: str, lang: str, session_id: int):
        """Asynchronous execution task. 

        Fenced at multiple boundaries to safely discard stale frames if a 
        higher session_id has been requested.
        """
        try:
            if not os.path.exists(SILVIE_VOICE_WAV):
                await websocket.send_text(json.dumps({
                    "type": "error", "message": f"Voice file missing: {SILVIE_VOICE_WAV}"
                }))
                return

            sentences = _clean_and_split_text(raw_text)
            tts_model = _get_tts()
            sent_count = 0

            for index, sentence in enumerate(sentences):
                # Check 1: Pre-generation session lock check
                if current_session_id != session_id:
                    logger.info(f"[WS/TTS] Session {session_id} superceded. Aborting synthesis chain.")
                    return

                try:
                    async with _gpu_semaphore:
                        # Check 2: Re-verify session validity right after waking up from the semaphore lock
                        if current_session_id != session_id: 
                            return

                        wav_bytes = await loop.run_in_executor(
                            None, _synth_chunk,
                            tts_model, sentence, SILVIE_VOICE_WAV, lang
                        )

                    # Check 3: Post-generation safety check before network transport
                    if current_session_id != session_id: 
                        return
                        
                    if not wav_bytes: 
                        continue

                    audio_b64 = base64.b64encode(wav_bytes).decode()
                    await websocket.send_text(json.dumps({
                        "type":  "audio_chunk",
                        "data":  audio_b64,
                        "index": index,
                        "total": len(sentences),
                    }))
                    sent_count += 1
                    logger.info(f"[WS/TTS] Sent chunk {index+1}/{len(sentences)} for session {session_id}")

                except Exception as chunk_exc:
                    logger.warning(f"[WS/TTS] Skipping unstable execution frame {index+1}: {chunk_exc}")
                    continue

            # Complete transaction signature check
            if current_session_id == session_id:
                await websocket.send_text(json.dumps({"type": "done"}))
                logger.info(f"[WS/TTS] Session {session_id} successfully finalized. Outflow complete.")

        except Exception as pipeline_err:
            logger.exception(f"[WS/TTS] Critical background synthesis loop failure: {pipeline_err}")
            try:
                await websocket.send_text(json.dumps({"type": "error", "message": "Internal loop crash"}))
            except Exception:
                pass

    # ── PERSISTENT READER LOOP ──
    try:
        while True:
            # Main socket loop remains open and unblocked to receive control commands immediately
            try:
                raw_msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
                continue

            try:
                msg = json.loads(raw_msg)
            except Exception:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            msg_type = msg.get("type")

            if msg_type == "ping" or msg.get("text") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            if msg_type == "cancel" or msg.get("text") == "cancel":
                logger.info("[WS/TTS] Intercepted instant explicit stop command.")
                current_session_id += 1  # Invalidation fence
                continue

            # Standard processing block
            text = msg.get("text", "").strip()
            language = msg.get("language", "de")
            lang_code = {"de": "de", "en": "en"}.get(language.lower(), "en")

            if not text:
                await websocket.send_text(json.dumps({"type": "error", "message": "Empty text submission"}))
                continue

            # Increment to invalidate any prior synthesis threads running down in process_synthesis
            current_session_id += 1 
            
            # Spin up a decoupled background task so the socket read loop continues listening instantly
            asyncio.create_task(
                process_synthesis(text, lang_code, current_session_id)
            )

    except WebSocketDisconnect:
        logger.info("[WS/TTS] Connection closed cleanly by client.")
    except Exception as exc:
        logger.exception(f"[WS/TTS] Global handling exception: {exc}")

# ── POST /session/clear ───────────────────────────────────────────────────────
@app.post("/session/clear", summary="Clear conversation history for a session")
def clear_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
    return {"status": "cleared", "session_id": session_id}


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
def health():
    from src.predict_v2 import _model_instance, _preload_done
    model_ready = _model_instance is not None
    return {
        "status":          "ok" if model_ready else "loading",
        "model":           "Silvie Llama 3.2 3B (QLoRA)",
        "model_ready":     model_ready,
        "languages":       ["de", "en"],
        "active_sessions": len(_sessions),
    }


# ── GET /metrics ──────────────────────────────────────────────────────────────
@app.get("/metrics", summary="Runtime statistics")
def metrics():
    from src.predict_v2 import _model_instance, _response_cache
    return {
        "uptime_seconds":  round(time.monotonic() - _server_start, 1),
        "model_loaded":    _model_instance is not None,
        "active_sessions": len(_sessions),
        "total_requests":  _request_count,
        "total_streams":   _stream_count,
        "cache_entries":   len(_response_cache),
        "languages":       ["de", "en"],
    }


# ── GET /warmup ───────────────────────────────────────────────────────────────
@app.get("/warmup", summary="Prime the model KV cache with a silent inference run")
async def warmup():
    from src.predict_v2 import _model_instance
    if _model_instance is None:
        return {"status": "model_not_ready"}

    loop = asyncio.get_event_loop()

    def _run():
        _llm_instance.create_chat_completion(
            messages   = [{"role": "user", "content": "Hi"}],
            max_tokens = 1,
            temperature= 0.0,
        )

    t0      = time.monotonic()
    await loop.run_in_executor(None, _run)
    elapsed = round(time.monotonic() - t0, 3)
    logger.info(f"[warmup] done in {elapsed}s")
    return {"status": "warm", "elapsed_seconds": elapsed}


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL TTS  (Coqui XTTS v2 — Silvie cloned voice)
# ═══════════════════════════════════════════════════════════════════════════════
SILVIE_VOICE_WAV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "voices", "silvie.wav"
)
_tts_instance = None
_tts_lock     = threading.Lock()
_tts_ready    = threading.Event()


def _get_tts():
    global _tts_instance
    if _tts_instance is not None:
        return _tts_instance
    with _tts_lock:
        if _tts_instance is not None:
            return _tts_instance
        from TTS.api import TTS as CoquiTTS
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        logger.info(f"[TTS] Loading XTTS v2 on {device}…")
        model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        _tts_instance = model
        logger.info("[TTS] ✅ XTTS v2 ready.")
    return _tts_instance


def _preload_tts():
    try:
        _get_tts()
    except Exception as exc:
        logger.warning(f"[TTS] Preload failed (non-fatal): {exc}")
    finally:
        _tts_ready.set()


class TTSRequest(BaseModel):
    text:     str
    language: Optional[str] = "de"


@app.post("/tts", summary="Silvie cloned voice TTS (XTTS v2, local GPU)")
async def text_to_speech(body: TTSRequest):
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="Text must not be empty.")
    if not os.path.exists(SILVIE_VOICE_WAV):
        raise HTTPException(
            status_code=503,
            detail=f"Voice file missing at {SILVIE_VOICE_WAV}",
        )
    text = body.text.strip()
    lang = {"de": "de", "en": "en"}.get((body.language or "de").lower(), "de")
    try:
        tts  = _get_tts()
        loop = asyncio.get_event_loop()

        def _synth():
            return tts.tts(
                text=text,
                speaker_wav=SILVIE_VOICE_WAV,
                language=lang,
            )

        wav = await loop.run_in_executor(None, _synth)

        # float32 list → 16-bit PCM WAV bytes
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)      # 16-bit
            wf.setframerate(24000)  # XTTS v2 native sample rate
            pcm = array.array("h", [
                max(-32768, min(32767, int(s * 32767))) for s in wav
            ])
            wf.writeframes(pcm.tobytes())
        buf.seek(0)
        return Response(
            content    = buf.read(),
            media_type = "audio/wav",
            headers    = {
                "Cache-Control":              "no-store",
                "Access-Control-Allow-Origin": "*",
            },
        )
    except Exception as exc:
        logger.exception("[TTS] Synthesis error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/tts/health", summary="TTS model status")
def tts_health():
    return {
        "status":       "ready" if _tts_instance is not None else "loading",
        "model":        "xtts_v2",
        "device":       "cuda" if _torch.cuda.is_available() else "cpu",
        "voice":        os.path.basename(SILVIE_VOICE_WAV),
        "voice_exists": os.path.exists(SILVIE_VOICE_WAV),
    }

# ── GET /scalar ───────────────────────────────────────────────────────────────
@app.get("/scalar", include_in_schema=False)
async def scalar_docs():
    try:
        from scalar_fastapi import get_scalar_api_reference
        return get_scalar_api_reference(
            openapi_url = app.openapi_url,
            title       = "Silvie AI API Documentation",
        )
    except ImportError:
        return {"message": "pip install scalar-fastapi for interactive docs"}
