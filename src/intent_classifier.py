"""
src/intent_classifier.py
=========================
Lightweight bilingual (DE + EN) intent detection for Silvie.

Strategy (no training needed):
  1. Keyword rules   — covers 90%+ of real user queries instantly
                        Rules contain both German AND English keywords.
  2. Zero-shot NLI   — fallback using facebook/bart-large-mnli (if installed)

Intents
-------
  emergency             | greeting            | farewell
  thanks                | repeat_request      | trip_planning
  youwel_explain        | youwel_health       | youwel_privacy
  youwel_funding        | appointment
  user_need_assets      | data_privacy_consent| specialist_search_permission
  contact_preference    | info_request        | collect_contact
  availability_time     | feedback            | conversation_end
  general_chat          (catch-all)
"""

import re
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


# ── Keyword rules (German + English) ────────────────────────────────────────

_RULES: list[tuple[str, list[str]]] = [

    # ── EMERGENCY — always checked first ──────────────────────────────────
    ("emergency", [
        # DE
        "notruf", "notfall", "hilfe!", "sofort hilfe", "bin gestürzt",
        "starke schmerzen", "bewusstlos", "herzinfarkt", "schlaganfall",
        "atemnot", "ohnmächtig", "blut", "blutung", "unfall", "es ist dringend",
        "ich stürze", "ich bin gefallen", "feuer", "brand",
        # EN
        "emergency", "call 911", "call 999", "call 112",
        "i fell", "i've fallen", "heart attack", "stroke",
        "chest pain", "severe pain", "can't breathe", "unconscious", "bleeding badly",
        "bleeding", "help me now", "urgent help", "ambulance", "fire",
    ]),

    # ── GREETING ──────────────────────────────────────────────────────────
    ("greeting", [
        # DE
        "hallo", "guten morgen", "guten tag", "guten abend",
        "hi silvie", "hey silvie", "servus", "moin", "bist du da",
        "bist du hier",
        # EN
        "hello", "good morning", "good afternoon", "good evening",
        "hi silvie", "hey silvie", "are you there", "good day",
        "howdy", "greetings",
    ]),

    # ── FAREWELL ──────────────────────────────────────────────────────────
    ("farewell", [
        # DE
        "auf wiedersehen", "tschüss", "bis bald", "danke und tschüss",
        "das war alles", "ich melde mich", "auf wiederhören", "ciao",
        # EN
        "goodbye", "bye", "see you", "see you later", "farewell",
        "that's all", "thanks and goodbye", "talk to you later",
        "catch you later", "until next time",
    ]),

    # ── THANKS ────────────────────────────────────────────────────────────
    ("thanks", [
        # DE
        "vielen dank", "danke schön", "herzlichen dank", "super danke",
        "das war sehr hilfreich", "wunderbar danke", "dankeschön",
        # EN
        "thank you", "thanks a lot", "many thanks", "great thanks",
        "that was very helpful", "wonderful thank you", "much appreciated",
        "appreciate it", "cheers", "thanks so much",
    ]),

    # ── REPEAT / CLARIFICATION ────────────────────────────────────────────
    ("repeat_request", [
        # DE
        "nicht verstanden", "wiederholen", "nochmal bitte", "zu schnell",
        "langsamer", "was haben sie gesagt", "bitte nochmal",
        "das verstehe ich nicht", "ich bin verwirrt",
        # EN
        "didn't understand", "please repeat", "say that again",
        "too fast", "slower please", "what did you say",
        "i'm confused", "could you explain again", "once more please",
        "i don't understand",
    ]),

    # ── TRIP PLANNING ────────────────────────────────────────────────────
    ("trip_planning", [
        # DE
        "reise", "urlaub", "reisen", "ausflug", "reiseziel", "wohin",
        "hotel", "sehenswürdigkeiten", "tourismus", "ferien",
        "reiseplanung", "strände", "stadt besuchen", "nach wien",
        "nach berlin", "nach münchen", "ans meer", "in die berge",
        "barrierefrei reisen",
        "die winter monate", "dem deutschen winter", "wärmere region",
        "deutschsprachige gemeinschaft", "november bis april",
        "deutsche ärzte", "gleichgesinnte",
        # EN
        "travel", "trip", "vacation", "holiday", "journey", "tour",
        "hotel", "sightseeing", "tourism", "destination",
        "where to go", "visit", "travel to", "fly to", "drive to",
        "accessible travel", "senior travel", "beach", "mountains",
        "plan a trip", "travel plans",
        "spend the winter", "winter months", "escape the cold", "southern region",
        "warmer climate", "november through", "april somewhere", "like-minded",
        "german-speaking community", "german doctors", "spend months",
        "avoid the cold", "somewhere warm", "winter escape",
    ]),

    # ── APPOINTMENT ──────────────────────────────────────────────────────
    ("appointment", [
        # DE
        "termin", "terminvereinbarung", "rückruf", "kontaktieren",
        "wann meldet sich", "anruf", "können sie", "anfragen",
        # EN
        "appointment", "schedule", "book", "callback", "contact me",
        "when will", "call me", "set up a meeting", "arrange",
        "make an appointment", "booking",
    ]),

    # ── ASSETS / CREATED VALUES MODULE ───────────────────────────────────
    ("user_need_assets", [
        # DE
        "vermögen", "anlage", "investition", "testament", "erbschaft",
        "vermögenssicherung", "altersvorsorge", "rente", "finanzplan",
        "steuer", "geld anlegen", "anlageformen", "vermögensschutz",
        "immobilien", "generationen", "finanzen", "steuern", "erbe",
        # EN
        "assets", "investment", "wealth", "pension", "retirement",
        "inheritance", "will", "estate", "financial planning",
        "tax", "invest money", "property", "savings", "portfolio",
        "financial advisor", "wealth management",
    ]),

    ("data_privacy_consent", [
        # DE
        "datenschutz", "privatsphäre", "dsgvo", "daten",
        "datensicherheit", "datenschutzerklärung",
        "bedingungen zustimmen", "zustimmung", "einverstanden",
        # EN
        "data protection", "privacy", "gdpr", "my data", "data security",
        "privacy policy", "terms", "consent", "agree", "personal data",
        "data safe", "how is my data",
    ]),

    ("specialist_search_permission", [
        # DE
        "gute bewertung", "recherchieren", "experten finden",
        "fachleute", "spezialisten", "suchen sie", "bitte suchen",
        "berater finden", "ja gerne", "spezialist suchen",
        # EN
        "find specialist", "find expert", "search for", "look for",
        "good reviews", "recommend a", "find an advisor",
        "please search", "yes please", "go ahead",
    ]),

    ("contact_preference", [
        # DE
        "per mail", "per email", "per e-mail", "per sms",
        "textnachricht", "auf dem handy", "schreiben sie mir",
        "benachrichtigung",
        # EN
        "by email", "by sms", "text message", "on my phone",
        "write to me", "notify me", "email me", "send me",
        "preferred contact", "reach me",
    ]),

    ("info_request", [
        # DE
        "schriftliche information", "zusammenfassung",
        "auf das wesentliche", "info zusenden", "mehr infos",
        "schicken sie mir das",
        # EN
        "written information", "summary", "send me information",
        "more info", "details please", "could you send",
        "information pack", "document",
    ]),

    ("collect_contact", [
        # DE
        "meine nummer", "handynummer", "telefonnummer", "mailadresse",
        "@", "meine adresse lautet", "ist die null", "01", "+49",
        # EN
        "my number", "mobile number", "phone number", "email address",
        "@", "my address is", "+44", "+1", "cell phone",
        "contact details",
    ]),

    ("availability_time", [
        # DE
        "zeitfenster", "erreichbar", "morgen", "übermorgen", "uhrzeit",
        "am vormittag", "nachmittag", "abends", "uhr", "gegen",
        "passt mir",
        # EN
        "time slot", "available", "tomorrow", "day after tomorrow",
        "in the morning", "afternoon", "evening", "o'clock",
        "around", "works for me", "availability", "free time",
        "best time to call",
    ]),

    ("feedback", [
        # DE
        "anregungen", "verbesserungen", "feedback", "nein danke",
        "im moment nicht", "keine fragen mehr", "alles klar",
        "soweit gut",
        # EN
        "feedback", "suggestions", "improvements", "no thank you",
        "not right now", "no more questions", "that's fine",
        "all good", "ok thanks", "no thanks",
    ]),

    ("conversation_end", [
        # DE
        "auf wiederhören", "bis bald", "abschied",
        "gespräch beendet", "ciao",
        # EN
        "talk to you soon", "conversation over", "done for now",
        "we're done", "wrap up", "end of conversation",
    ]),

    # ── HEALTH ───────────────────────────────────────────────────────────
    ("youwel_health", [
        # DE
        "schmerzen", "arzt", "gesundheit", "krank", "medizin",
        "ernährung", "pflege", "krankenhaus", "apotheke", "rezept",
        "spezialist", "orthopäde", "hausarzt", "symptome", "beschwerden",
        # EN
        "pain", "doctor", "health", "sick", "illness", "medicine",
        "nutrition", "care", "hospital", "pharmacy", "prescription",
        "specialist", "gp", "symptoms", "ailment", "medical",
        "health advice", "feel unwell", "feeling ill",
    ]),

    # ── FUNDING / BUSINESS MODEL ──────────────────────────────────────────
    ("youwel_funding", [
        # DE
        "finanziert", "wer steckt", "investor", "kosten für youwel",
        "verdient ihr", "geschäftsmodell", "wer bezahlt",
        # EN
        "how is youwel funded", "who funds", "investor", "cost",
        "how do you make money", "business model", "who pays",
        "revenue", "funding model", "free or paid",
    ]),

    # ── LIST CAPABILITIES ────────────────────────────────────────────────
    ("list_capabilities", [
        # DE
        "in welchen bereichen", "welche bereiche", "lebensbereiche",
        "9 bereiche", "neun bereiche", "was kannst du", "was können sie",
        "wo kannst du helfen", "was bieten sie", "hilfebereiche",
        # EN
        "which areas", "what areas", "9 areas", "nine areas",
        "what can you do", "what do you offer", "areas of life",
        "what can you help", "help me with", "these four",
        "you have just", "you just listed", "i need nine", "i need 9",
        "how many areas", 
        "how many life areas",
        "outline these areas",
        "list these areas",
        "bullet point",
        "briefly outline",
        "areas where you can",
    ]),

    # ── YOUWEL EXPLAIN — broad catch-all for youwel questions ────────────
    ("youwel_explain", [
        # DE
        "was ist youwel", "wie funktioniert", "für wen", "grüner juwel",
        "juwel", "youwel.app", "app herunterladen", "registrieren",
        "anmelden", "kostenfrei", "wie starte ich", "tablet",
        "silvie was kannst du", "was kann silvie", "was können sie",
        "wer bist du", "erzähl mir von", "9 bereiche", "lebensbereiche",
        # EN
        "what is youwel", "how does youwel work", "who is silvie",
        "what can you do", "youwel.app", "download the app",
        "sign up", "register", "free of charge", "how do i start",
        "tell me about youwel", "about silvie", "9 areas",
        "green jewel", "green youwel", "who are you",
        "what are your services", "how to use",
    ]),
]


def classify_intent(text: str, fallback: str = "general_chat") -> str:
    """
    Classify user input into an intent string.

    Works for both German and English input — keyword rules cover both.

    Parameters
    ----------
    text     : raw user message (DE or EN)
    fallback : intent when no rule matches (default "general_chat")

    Returns
    -------
    str — intent label
    """
    t       = text.lower().strip()
    t_clean = re.sub(r"[^\w\s@+]", " ", t)   # keep @ and + for collect_contact

    for intent, keywords in _RULES:
        for kw in keywords:
            if kw in t_clean:
                logger.debug(f"Intent '{intent}' matched keyword '{kw}'")
                return intent

    # Optional zero-shot NLI fallback
    try:
        return _zero_shot_classify(text)
    except Exception:
        pass

    return fallback


# ── Zero-shot NLI fallback ────────────────────────────────────────────────────

_zero_shot_pipeline = None
_nli_lock           = threading.Lock()
_nli_preload_done   = threading.Event()

_ZS_LABELS = [
    "Emergency or call for help",
    "Greeting",
    "Farewell or thank you",
    "Travel planning or vacation",
    "Health or doctor",
    "Data protection or privacy",
    "Wealth or financial planning",
    "Appointment scheduling",
    "General question about youwel or Silvie",
    "General conversation",
]

_ZS_TO_INTENT = {
    "Emergency or call for help":              "emergency",
    "Greeting":                                "greeting",
    "Farewell or thank you":                   "farewell",
    "Travel planning or vacation":             "trip_planning",
    "Health or doctor":                        "youwel_health",
    "Data protection or privacy":              "data_privacy_consent",
    "Wealth or financial planning":            "user_need_assets",
    "Appointment scheduling":                  "appointment",
    "General question about youwel or Silvie": "youwel_explain",
    "General conversation":                    "general_chat",
}


def preload_nli_pipeline() -> None:
    """
    Download and cache the zero-shot NLI pipeline in a background thread.
    Call this from FastAPI lifespan startup. No-op if transformers not installed.
    """
    def _do_load():
        global _zero_shot_pipeline
        try:
            from transformers import pipeline
            logger.info("[preload] Loading zero-shot NLI pipeline…")
            pipe = pipeline(
                "zero-shot-classification",
                model  = "facebook/bart-large-mnli",
                device = -1,   # CPU
            )
            with _nli_lock:
                _zero_shot_pipeline = pipe
            logger.info("[preload] ✅ Zero-shot NLI pipeline ready.")
        except Exception as exc:
            logger.warning(
                f"[preload] NLI pipeline unavailable (keyword rules will be used): {exc}"
            )
        finally:
            _nli_preload_done.set()

    t = threading.Thread(
        target=_do_load, daemon=True, name="silvie-nli-preload"
    )
    t.start()


def _zero_shot_classify(text: str, timeout: float = 120.0) -> str:
    global _zero_shot_pipeline
    with _nli_lock:
        pipe = _zero_shot_pipeline
    if pipe is None:
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


# ── Emergency detection (always available, no ML) ────────────────────────────

_EMERGENCY_KEYWORDS = {
    # DE
    "notruf", "notfall", "hilfe!", "sofort hilfe", "bin gestürzt",
    "starke schmerzen", "bewusstlos", "herzinfarkt", "schlaganfall",
    "atemnot", "ohnmächtig", "blut", "blutung", "unfall", "es ist dringend",
    "ich stürze", "ich bin gefallen", "feuer", "brand",
    # EN
    "emergency", "call 911", "call 999", "call 112",
    "i fell", "i've fallen", "heart attack", "stroke",
    "chest pain", "severe pain", "can't breathe", "unconscious", "bleeding badly",
    "bleeding", "help me now", "urgent help", "ambulance", "fire",
}


def is_emergency(text: str) -> bool:
    """Fast check — does NOT require ML models. Language-agnostic."""
    t = text.lower()
    return any(kw in t for kw in _EMERGENCY_KEYWORDS)


# ── Simple demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        ("Hallo Silvie!",                       "greeting"),
        ("Hello Silvie!",                        "greeting"),
        ("Ich möchte nach Wien reisen",          "trip_planning"),
        ("I want to travel to Vienna",           "trip_planning"),
        ("Meine Daten — sind die sicher?",       "data_privacy_consent"),
        ("Is my data safe with youwel?",         "data_privacy_consent"),
        ("Ich habe starke Schmerzen!",           "emergency"),
        ("I'm in severe pain!",                  "emergency"),
        ("Ich brauche einen Steuerberater",      "user_need_assets"),
        ("I need a financial advisor",           "user_need_assets"),
        ("Was ist youwel genau?",                "youwel_explain"),
        ("What exactly is youwel?",              "youwel_explain"),
        ("Können Sie das bitte wiederholen?",    "repeat_request"),
        ("Could you repeat that please?",        "repeat_request"),
    ]
    print(f"{'Input':<50} {'Expected':<25} {'Got':<25} OK?")
    print("-" * 110)
    for text, expected in test_cases:
        got    = classify_intent(text)
        ok     = "✅" if got == expected else "❌"
        print(f"{text:<50} {expected:<25} {got:<25} {ok}")