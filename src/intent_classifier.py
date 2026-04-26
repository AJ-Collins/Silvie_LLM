"""
src/intent_classifier.py
=========================
Lightweight intent detection for Silvie.

Strategy (no training needed):
  1. Keyword rules   — covers 90%+ of real user queries instantly
  2. Zero-shot NLI   — fallback using facebook/bart-large-mnli (if installed)

Intents:
  emergency       | greeting      | farewell     | thanks
  repeat_request  | trip_planning | youwel_assets| youwel_health
  youwel_privacy  | youwel_explain| youwel_funding| appointment
  general_chat    (catch-all)
"""

import re
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Keyword rules (German)

_RULES: list[tuple[str, list[str]]] = [
    # EMERGENCY — highest priority, always checked first
    ("emergency", [
        "notruf", "notfall", "hilfe!", "sofort hilfe", "bin gestürzt",
        "starke schmerzen", "bewusstlos", "herzinfarkt", "schlaganfall",
        "atemnot", "ohnmächtig", "blut", "unfall", "es ist dringend",
    ]),
    # GREETINGS
    ("greeting", [
        "hallo", "guten morgen", "guten tag", "guten abend", "hi silvie",
        "hey silvie", "guten tag silvie", "hallo silvie", "servus",
        "moin", "morgen", "bist du da", "bist du hier",
    ]),
    # FAREWELL
    ("farewell", [
        "auf wiedersehen", "tschüss", "bis bald", "danke und tschüss",
        "das war alles", "ich melde mich", "auf wiederhören", "ciao",
    ]),
    # THANKS
    ("thanks", [
        "vielen dank", "danke schön", "herzlichen dank", "super danke",
        "das war sehr hilfreich", "wunderbar danke",
    ]),
    # REPEAT
    ("repeat_request", [
        "nicht verstanden", "wiederholen", "nochmal bitte", "zu schnell",
        "langsamer", "was haben sie gesagt", "bitte nochmal",
        "das verstehe ich nicht", "ich bin verwirrt",
    ]),
    # TRIP PLANNING
    ("trip_planning", [
        "reise", "urlaub", "reisen", "ausflug", "reiseziel", "wohin",
        "hotel", "sehenswürdigkeiten", "tourismus", "ferien", "reiseplanung",
        "strände", "stadt besuchen", "nach wien", "nach berlin", "nach münchen",
        "ans meer", "in die berge", "barrierefrei reisen",
    ]),
    # APPOINTMENT
    ("appointment", [
        "termin", "terminvereinbarung", "rückruf", "kontaktieren",
        "wann meldet sich", "anruf", "können sie", "anfragen",
    ]),
    
    # MODULE: GESCHAFFENE WERTE (Assets Scripted Flow)
    ("user_need_assets", [
        "vermögen", "anlage", "investition", "testament", "erbschaft", "vermögenssicherung",
        "altersvorsorge", "rente", "finanzplan", "steuer", "geld anlegen", "anlageformen",
        "vermögensschutz", "immobilien", "generationen", "finanzen", "steuern", "erbe",
    ]),
    ("data_privacy_consent", [
        "datenschutz", "privatsphäre", "dsgvo", "daten", "datensicherheit", 
        "datenschutzerklärung", "bedingungen zustimmen", "zustimmung", "einverstanden",
    ]),
    ("specialist_search_permission", [
        "gute bewertung", "recherchieren", "experten finden", "fachleute", "spezialisten", 
        "suchen sie", "bitte suchen", "berater finden", "ja gerne", "spezialist suchen",
    ]),
    ("contact_preference", [
        "per mail", "per email", "per e-mail", "per sms", "textnachricht", 
        "auf dem handy", "schreiben sie mir", "benachrichtigung",
    ]),
    ("info_request", [
        "schriftliche information", "zusammenfassung", "auf das wesentliche", 
        "info zusenden", "mehr infos", "schicken sie mir das",
    ]),
    ("collect_contact", [
        "meine nummer", "handynummer", "telefonnummer", "mailadresse", "@",
        "meine adresse lautet", "ist die null", "01", "+49",
    ]),
    ("availability_time", [
        "zeitfenster", "erreichbar", "morgen", "übermorgen", "uhrzeit",
        "am vormittag", "nachmittag", "abends", "uhr", "gegen", "passt mir",
    ]),
    ("feedback", [
        "anregungen", "verbesserungen", "feedback", "nein danke", "im moment nicht",
        "keine fragen mehr", "alles klar", "soweit gut",
    ]),
    ("conversation_end", [
        "auf wiederhören", "bis bald", "abschied", "gespräch beendet", "ciao",
    ]),
    # HEALTH
    ("youwel_health", [
        "schmerzen", "arzt", "gesundheit", "krank", "medizin", "ernährung",
        "pflege", "krankenhaus", "apotheke", "rezept", "spezialist",
        "orthopäde", "hausarzt", "symptome", "beschwerden",
    ]),
    # FUNDING
    ("youwel_funding", [
        "finanziert", "wer steckt", "investor", "kosten für youwel",
        "verdient ihr", "geschäftsmodell", "wer bezahlt",
    ]),
    # YOUWEL EXPLAIN (broad — catch-all for youwel questions)
    ("youwel_explain", [
        "was ist youwel", "wie funktioniert", "für wen", "grüner juwel",
        "juwel", "youwel.app", "app herunterladen", "registrieren",
        "anmelden", "kostenfrei", "wie starte ich", "tablet",
        "silvie was kannst du", "was kann silvie", "was können sie",
    ]),
]


def classify_intent(text: str, fallback: str = "general_chat") -> str:
    """
    Classify user input into an intent string.

    Parameters
    ----------
    text     : raw user message
    fallback : intent to return when no rule matches

    Returns
    -------
    str  — intent label
    """
    t = text.lower().strip()
    # Remove punctuation for matching
    t_clean = re.sub(r"[^\w\s]", " ", t)

    for intent, keywords in _RULES:
        for kw in keywords:
            if kw in t_clean:
                logger.debug(f"Intent '{intent}' matched keyword '{kw}'")
                return intent

    # Optional: zero-shot NLI fallback (requires transformers)
    try:
        return _zero_shot_classify(text)
    except Exception:
        pass

    return fallback


# Zero-shot fallback — eager preloading

_zero_shot_pipeline  = None
_nli_lock            = threading.Lock()
_nli_preload_done    = threading.Event()

_ZS_LABELS = [
    "Notfall oder Hilferuf",
    "Begrüßung",
    "Abschied oder Danke",
    "Reiseplanung oder Urlaub",
    "Gesundheit oder Arzt",
    "Datenschutz oder Datensicherheit",
    "Vermögen oder Finanzen",
    "Terminvereinbarung",
    "Allgemeine Frage zu youwel",
    "Allgemeines Gespräch",
]

_ZS_TO_INTENT = {
    "Notfall oder Hilferuf":              "emergency",
    "Begrüßung":                          "greeting",
    "Abschied oder Danke":                "farewell",
    "Reiseplanung oder Urlaub":           "trip_planning",
    "Gesundheit oder Arzt":               "youwel_health",
    "Datenschutz oder Datensicherheit":   "data_privacy_consent",
    "Vermögen oder Finanzen":             "user_need_assets",
    "Terminvereinbarung":                 "appointment",
    "Allgemeine Frage zu youwel":         "youwel_explain",
    "Allgemeines Gespräch":               "general_chat",
}


def preload_nli_pipeline() -> None:
    """
    Download & cache the zero-shot NLI pipeline in a background thread.
    Call this from FastAPI lifespan startup so the model is ready before
    the first user request — no download latency mid-conversation.

    If transformers is not installed this is a no-op (keyword rules still work).
    """
    def _do_load():
        global _zero_shot_pipeline
        try:
            from transformers import pipeline
            logger.info("[preload] Loading zero-shot NLI pipeline…")
            pipe = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,   # CPU
            )
            with _nli_lock:
                _zero_shot_pipeline = pipe
            logger.info("[preload] ✅ Zero-shot NLI pipeline ready.")
        except Exception as exc:
            logger.warning(f"[preload] NLI pipeline unavailable (keyword rules will be used): {exc}")
        finally:
            _nli_preload_done.set()

    t = threading.Thread(target=_do_load, daemon=True, name="silvie-nli-preload")
    t.start()


def _zero_shot_classify(text: str, timeout: float = 120.0) -> str:
    """Use the preloaded NLI pipeline; waits up to `timeout` seconds for preload."""
    global _zero_shot_pipeline

    # Fast path: already loaded
    with _nli_lock:
        pipe = _zero_shot_pipeline
    if pipe is None:
        # Wait for background preload (won't block if preload_nli_pipeline was called)
        _nli_preload_done.wait(timeout=timeout)
        with _nli_lock:
            pipe = _zero_shot_pipeline
    if pipe is None:
        raise RuntimeError("NLI pipeline not available")

    result    = pipe(text, candidate_labels=_ZS_LABELS)
    top_label = result["labels"][0]
    intent    = _ZS_TO_INTENT.get(top_label, "general_chat")
    logger.debug(f"Zero-shot → '{top_label}' → intent '{intent}'")
    return intent


# Emergency detection (always available, independent)

_EMERGENCY_KEYWORDS = {
    "notruf", "notfall", "hilfe!", "herzinfarkt", "schlaganfall",
    "bewusstlos", "bin gestürzt", "starke schmerzen", "blutung",
    "atemnot", "ohnmächtig", "unfall", "feuer", "brand",
}

def is_emergency(text: str) -> bool:
    """Fast check — does NOT require ML models."""
    t = text.lower()
    return any(kw in t for kw in _EMERGENCY_KEYWORDS)


# Simple demo
if __name__ == "__main__":
    test_cases = [
        "Hallo Silvie!",
        "Ich möchte nach Wien reisen",
        "Meine Daten — sind die sicher?",
        "Ich habe starke Schmerzen!",
        "Ich brauche einen Steuerberater",
        "Was ist youwel genau?",
        "Können Sie das bitte wiederholen?",
    ]
    for t in test_cases:
        intent = classify_intent(t)
        print(f"  [{intent:20s}]  {t}")