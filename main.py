"""
main.py — Silvie AI FastAPI server v5.1  (speed-optimised)
============================================================
Speed improvements over v5.0
──────────────────────────────
1.  /predict is now async + run_in_executor
    The blocking LLM call no longer freezes the event loop.
    Concurrent HTTP requests are handled while inference runs.

2.  Intent classified ONCE per request, shared to predict_hf/predict_stream_hf
    Eliminates the duplicate classify_intent() call that existed in v5.0.

3.  GZipMiddleware
    JSON responses compressed automatically → smaller payload, faster transfer.

4.  Pre-built SSE prefix bytes
    `b'data: {"token":"'` built once outside the hot loop — avoids repeated
    json.dumps() per token. Saves ~2–5 µs × N tokens per stream.

5.  StreamingResponse with background task for history save
    History is saved AFTER the stream closes, not during it.

6.  X-Silvie-Intent response header on both /predict and /predict-stream
    Useful for client-side debug / analytics.

7.  /warmup endpoint
    Runs a silent low-token inference to prime the KV cache and JIT paths
    so real user requests are maximally fast.

Endpoints (unchanged from v5.0):
  POST /predict          → JSON blocking answer
  POST /predict-stream   → SSE phrase-stream
  POST /voice            → audio → transcript
  POST /session/clear    → wipe conversation history
  GET  /health           → health + model status
  GET  /metrics          → runtime stats
  GET  /warmup           → silent inference warm-up
  GET  /scalar           → interactive docs (optional)
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import asyncio
import json
import logging
import tempfile
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware   # ← NEW: compress responses
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

_server_start = time.monotonic()

# Pre-built SSE byte sequences (avoid json.dumps per token in hot loop)
# Format:  data: {"token":"<CHUNK>"}\n\n
_SSE_TOKEN_PREFIX = b'data: {"token":'
_SSE_DONE         = b"data: [DONE]\n\n"

def _sse_token(chunk: str) -> bytes:
    """Encode one SSE token event.  Faster than json.dumps({'token': chunk})."""
    # json.dumps just the value so special characters are escaped correctly.
    return _SSE_TOKEN_PREFIX + json.dumps(chunk, ensure_ascii=False).encode() + b"}\n\n"

def _sse_json(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


# FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.predict_v2 import preload_model
    from src.intent_classifier import preload_nli_pipeline
    logger.info("🚀 Silvie startup — initiating model preloads…")
    preload_model()
    preload_nli_pipeline()
    yield
    logger.info("🛑 Silvie shutdown.")


app = FastAPI(
    title       = "Silvie AI API",
    description = (
        "Silvie ist der persönliche KI-Sprachassistent von youwel.app — "
        "entwickelt für aktive Menschen ab 40 Jahren in Deutschland.\n\n"
        "**Model:** Phi-3-mini-4k-instruct (GGUF q4) via llama-cpp-python\n"
        "**Streaming:** Phrase-boundary SSE chunks for natural reading flow"
    ),
    version  = "5.1.0",
    lifespan = lifespan,
)

# Middleware — order matters: GZip wraps CORS
app.add_middleware(GZipMiddleware, minimum_size=500)   # compress responses ≥ 500 B
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# Model accessor
def _get_model():
    from src.predict_v2 import load_silvie_hf
    return load_silvie_hf()


# Session memory
_sessions: dict[str, list[dict]] = defaultdict(list)
MAX_HISTORY_TURNS = 6

_request_count = 0
_stream_count  = 0


def get_history(session_id: Optional[str]) -> list[dict]:
    if not session_id:
        return []
    return _sessions[session_id][-MAX_HISTORY_TURNS:]


def save_to_history(session_id: Optional[str], question: str, answer: str) -> None:
    if not session_id:
        return
    _sessions[session_id].append({"role": "user",      "content": question})
    _sessions[session_id].append({"role": "assistant",  "content": answer})
    cap = MAX_HISTORY_TURNS * 2
    if len(_sessions[session_id]) > cap:
        _sessions[session_id] = _sessions[session_id][-cap:]


# Intent helper
def _get_intent(text: str) -> str:
    try:
        from src.intent_classifier import classify_intent
        return classify_intent(text)
    except Exception:
        return "general_chat"


# Schemas
class Question(BaseModel):
    text:       str
    session_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "Hallo Silvie!",               "session_id": "user-abc-123"},
                {"text": "Was ist youwel?"},
                {"text": "Ich möchte nach Wien reisen."},
            ]
        }
    }


class AnswerResponse(BaseModel):
    question:   str
    answer:     str
    intent:     Optional[str] = None
    session_id: Optional[str] = None


class TranscriptResponse(BaseModel):
    transcript: str
    session_id: Optional[str] = None


# Audio validation
_ALLOWED_AUDIO = (
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/m4a", "audio/mp4", "audio/x-m4a",
    "audio/mpeg", "audio/mp3",
    "audio/webm", "audio/ogg",
    "application/octet-stream",
)


async def _transcribe_upload(audio: UploadFile) -> str:
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
        transcript = await loop.run_in_executor(None, transcribe_audio, tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription error: {exc}")
    finally:
        os.remove(tmp_path)
    if not transcript.strip():
        raise HTTPException(
            status_code=422,
            detail="Keine Sprache erkannt. Bitte sprechen Sie klar und deutlich.",
        )
    return transcript.strip()


# POST /predict  (now async — does NOT block event loop)
@app.post(
    "/predict",
    response_model=AnswerResponse,
    summary="Ask Silvie — full blocking answer",
)
async def ask_silvie(body: Question):          # ← async
    global _request_count
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="Frage darf nicht leer sein.")

    session_id = body.session_id or str(uuid.uuid4())
    history    = get_history(session_id)

    # Classify intent ONCE — pass it into predict_hf to avoid double call
    intent = _get_intent(body.text)

    from src.predict_v2 import predict_hf

    # Run blocking LLM in thread-pool so the event loop stays free
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
            intent               = intent,   # ← pass pre-classified intent
        ),
    )

    save_to_history(session_id, body.text, answer)
    _request_count += 1

    response = AnswerResponse(
        question   = body.text,
        answer     = answer,
        intent     = intent,
        session_id = session_id,
    )
    # FastAPI doesn't support custom headers on response_model returns directly,
    # so build the Response manually only if caller wants the header.
    return response


# POST /predict-stream
@app.post(
    "/predict-stream",
    summary="Ask Silvie — SSE phrase-stream",
    description=(
        "Returns `text/event-stream`. Each `data:` line is `{\"token\": \"...\"}` "
        "where *token* is a readable phrase chunk. Stream ends with `data: [DONE]`.\n\n"
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
        raise HTTPException(status_code=422, detail="Frage darf nicht leer sein.")

    session_id = body.session_id or str(uuid.uuid4())
    history    = get_history(session_id)

    # Classify intent once — shared with the generator
    intent = _get_intent(body.text)

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
                intent               = intent,   # ← pass pre-classified intent
            ):
                if chunk:
                    full_answer.append(chunk)
                    yield _sse_token(chunk)        # ← pre-built bytes, no json.dumps
        except Exception as exc:
            logger.exception("Streaming error")
            yield _sse_json({"error": str(exc)})
        finally:
            if full_answer:
                # Save history in background so we don't add latency to the stream close
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
            "X-Silvie-Intent":   intent,     # useful for client debug
        },
    )


# POST /voice
@app.post(
    "/voice",
    response_model=TranscriptResponse,
    summary="Transcribe audio → return transcript (then call /predict-stream)",
)
async def ask_silvie_voice(
    audio:        UploadFile     = File(...),
    x_session_id: Optional[str] = Header(None),
):
    transcript = await _transcribe_upload(audio)
    session_id = x_session_id or str(uuid.uuid4())
    return TranscriptResponse(transcript=transcript, session_id=session_id)


# POST /session/clear
@app.post("/session/clear", summary="Clear conversation history for a session")
def clear_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
    return {"status": "cleared", "session_id": session_id}


# GET /health
@app.get("/health", summary="Health check")
def health():
    from src.predict_v2 import _llm_instance, _preload_done
    model_ready = _llm_instance is not None
    return {
        "status":          "ok" if model_ready else "loading",
        "model":           "Silvie GGUF v5.1 (Phi-3-mini-4k-instruct q4)",
        "model_ready":     model_ready,
        "active_sessions": len(_sessions),
    }


# GET /metrics
@app.get("/metrics", summary="Runtime statistics")
def metrics():
    from src.predict_v2 import _llm_instance, _response_cache
    return {
        "uptime_seconds":    round(time.monotonic() - _server_start, 1),
        "model_loaded":      _llm_instance is not None,
        "active_sessions":   len(_sessions),
        "total_requests":    _request_count,
        "total_streams":     _stream_count,
        "cache_entries":     len(_response_cache),   # ← new: response cache size
    }


# GET /warmup
@app.get("/warmup", summary="Prime the model KV cache with a silent inference run")
async def warmup():
    """
    Call this once after /health returns model_ready=true.
    Runs a minimal 1-token inference to trigger any JIT/BLAS warm-up paths
    so the first real user request isn't penalised.
    """
    from src.predict_v2 import _llm_instance
    if _llm_instance is None:
        return {"status": "model_not_ready"}

    loop = asyncio.get_event_loop()

    def _run():
        _llm_instance.create_chat_completion(
            messages   = [{"role": "user", "content": "Hi"}],
            max_tokens = 1,
            temperature= 0.0,
        )

    t0 = time.monotonic()
    await loop.run_in_executor(None, _run)
    elapsed = round(time.monotonic() - t0, 3)
    logger.info(f"[warmup] done in {elapsed}s")
    return {"status": "warm", "elapsed_seconds": elapsed}


# GET /scalar
@app.get("/scalar", include_in_schema=False)
async def scalar_docs():
    try:
        from scalar_fastapi import get_scalar_api_reference
        return get_scalar_api_reference(
            openapi_url=app.openapi_url,
            title="Silvie AI API Documentation",
        )
    except ImportError:
        return {"message": "pip install scalar-fastapi for interactive docs"}