"""
src/senior_filter.py
=====================
Post-process LLM responses to be senior-friendly.

Bilingual (DE + EN) — language selected by the `language` parameter
passed into format_for_senior().

Steps applied to every response:
  1. Replace jargon with plain-language equivalents
  2. Break overly long sentences at natural conjunctions
  3. Truncate to max 5 sentences
  4. Optionally append a follow-up offer
  5. Special handling for confusion / frustration signals
"""

import re
import logging
import unicodedata

logger = logging.getLogger(__name__)

MAX_SENTENCE_WORDS = 22


_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002600-\U000027BF"  # misc symbols
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U00002700-\U000027BF"  # dingbats
    "\U0001FA00-\U0001FA6F"  # chess symbols etc
    "\U0001FA70-\U0001FAFF"  # more symbols
    "]+",
    flags=re.UNICODE,
)
# ── Jargon maps ───────────────────────────────────────────────────────────────

_JARGON_MAP_DE: dict[str, str] = {
    "KI":           "Künstliche Intelligenz",
    "AI":           "Künstliche Intelligenz",
    "API":          "unsere digitale Verbindung",
    "DSGVO":        "deutsches Datenschutzgesetz",
    "QR-Code":      "QR-Code (das quadratische Bild zum Scannen)",
    "Browser":      "Internet-Programm",
    "Upload":       "Hochladen",
    "Download":     "Herunterladen",
    "SMS":          "Textnachricht",
    "Link":         "Internetadresse",
    "Server":       "Computer-Server",
    "Account":      "Benutzerkonto",
    "Password":     "Passwort",
    "Login":        "Anmeldung",
    "Update":       "Aktualisierung",
    "App":          "Anwendung",
    "E-Mail":       "E-Mail",   # already simple — keep
    "GDPR":         "europäisches Datenschutzgesetz",
}

_JARGON_MAP_EN: dict[str, str] = {
    "AI":           "Artificial Intelligence",
    "API":          "our digital connection",
    "GDPR":         "European data protection law",
    "DSGVO":        "German data protection law",
    "QR code":      "QR code (the square image you scan)",
    "browser":      "internet programme",
    "upload":       "send to the system",
    "download":     "save to your device",
    "SMS":          "text message",
    "link":         "web address",
    "server":       "computer server",
    "account":      "user account",
    "password":     "password",    # already simple
    "login":        "sign in",
    "update":       "refresh",
    "app":          "application",
}

# ── Confusion / frustration signals ──────────────────────────────────────────

_CONFUSION_DE = {
    "verstehe nicht", "was meinen sie", "ich bin verwirrt",
    "das ergibt keinen sinn", "zu kompliziert", "nicht verstanden",
    "bitte wiederholen", "zu schnell", "was haben sie gesagt",
    "ich komme nicht weiter", "ich weiß nicht was",
}

_FRUSTRATION_DE = {
    "das ist frustrierend", "ich bin genervt", "das klappt nicht",
    "nichts funktioniert", "ich verstehe das nicht mehr",
    "ich brauche einen menschen", "echten mitarbeiter",
    "ich will mit jemandem sprechen",
}

_CONFUSION_EN = {
    "don't understand", "didn't understand", "i'm confused",
    "makes no sense", "too complicated", "could you repeat",
    "please repeat", "too fast", "what did you say",
    "i'm lost", "i don't know what",
}

_FRUSTRATION_EN = {
    "this is frustrating", "i'm frustrated", "nothing works",
    "it's not working", "i can't figure", "i need a human",
    "real person please", "speak to someone",
    "i want to talk to someone",
}

# ── Sentence-split conjunctions ───────────────────────────────────────────────
_SPLIT_WORDS_DE = {"und", "aber", "oder", "damit", "weil", "denn", "jedoch"}
_SPLIT_WORDS_EN = {"and", "but", "or", "because", "so", "yet", "however"}


def is_confused(user_text: str, language: str = "de") -> bool:
    t = user_text.lower()
    signals = _CONFUSION_DE if language == "de" else _CONFUSION_EN
    return any(s in t for s in signals)


def is_frustrated(user_text: str, language: str = "de") -> bool:
    t = user_text.lower()
    signals = _FRUSTRATION_DE if language == "de" else _FRUSTRATION_EN
    return any(s in t for s in signals)


def replace_jargon(text: str, language: str = "de") -> str:
    """Replace technical terms with plain-language equivalents."""
    jargon_map = _JARGON_MAP_DE if language == "de" else _JARGON_MAP_EN
    for jargon, replacement in jargon_map.items():
        text = re.sub(
            rf"\b{re.escape(jargon)}\b", replacement, text, flags=re.IGNORECASE
        )
    return text


def shorten_sentences(
    text: str, max_words: int = MAX_SENTENCE_WORDS, language: str = "de"
) -> str:
    split_words = _SPLIT_WORDS_DE if language == "de" else _SPLIT_WORDS_EN

    # Preserve numbered/bulleted lists — don't flatten them
    if re.search(r'^\s*\d+\.', text, re.MULTILINE):
        return text   # ← return as-is; truncation is handled separately

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    result = []

    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            result.append(sentence)
            continue

        split_idx = None
        for i, w in enumerate(words[8:max_words], start=8):
            if w.lower() in split_words:
                split_idx = i
                break

        if split_idx:
            part1 = " ".join(words[:split_idx])
            part2 = " ".join(words[split_idx:])
            part2 = part2[0].upper() + part2[1:]
            result.append(part1 + ".")
            result.append(part2)
        else:
            result.append(" ".join(words[:max_words]) + "…")

    return " ".join(result)


def truncate_response(text: str, max_sentences: int = 5) -> str:
    """Keep only the first max_sentences sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= max_sentences:
        return text
    truncated = " ".join(sentences[:max_sentences])
    logger.debug(f"Truncated response from {len(sentences)} to {max_sentences} sentences")
    return truncated

_HALLUCINATION_PHRASES = [    
    "nine years",
    "career goals",
    "personal relationships",
    "holistic consulting",
    "life coach",
    "past nine",
    "overall sense of well-being",
    "improve their personal",
    "boost their",
    "personal development",
    "career guidance",
    "job search",    
    # German
    "neun jahre",
    "neun jahren",
    "karriereziele",
    "persönliche beziehungen",
    "ganzheitliche beratung",
    "lebensberater",
]

def _strip_hallucinations(text: str) -> str:
    """Return empty string if response contains known hallucinated phrases."""
    t_lower = text.lower()
    for phrase in _HALLUCINATION_PHRASES:
        if phrase in t_lower:
            logger.warning(f"[senior_filter] Hallucination detected: '{phrase}' — blanking response")
            return ""  # caller should substitute a canned response
    return text
    

def validate_areas_response(text: str, intent: str = "") -> str:
    """Ensure all 9 areas are present when listing capabilities."""
    # Count numbered items
    numbers_found = len(re.findall(r'\d+\.', text))
    
    if intent == "list_capabilities":
        if numbers_found < 9:
            logger.warning(f"Response only contained {numbers_found} areas for list_capabilities — incomplete")
            return ""  # triggers fallback
        return text

    if "1." not in text:
        return text
    
    if 1 <= numbers_found < 9:
        logger.warning(f"Response only contained {numbers_found} areas — incomplete")
        return ""  # triggers fallback
    return text


def add_followup_offer(text: str, language: str = "de") -> str:
    """Append a gentle follow-up offer if not already present."""
    # Don't add a second question if text already ends with one
    if text.rstrip().endswith("?"):
        return text

    if language == "en":
        follow_up_phrases = [
            "Can I", "Do you have", "Would you like", "Shall I",
            "May I", "Is there anything",
        ]
        follow_up = "Do you have any further questions?"
    else:
        follow_up_phrases = [
            "Kann ich", "Haben Sie noch", "Möchten Sie", "Soll ich",
            "Darf ich", "Wünschen Sie",
        ]
        follow_up = "Haben Sie noch weitere Fragen?"

    if not any(p in text for p in follow_up_phrases):
        text = text.rstrip(".!") + ". " + follow_up

    return text


def format_for_senior(
    response:     str,
    user_text:    str  = "",
    intent:       str  = "",
    language:     str  = "de",    # ← NEW param (defaults to "de" for backwards compat)
    add_followup: bool = True,
) -> str:
    """
    Full senior-friendly formatting pipeline.

    Parameters
    ----------
    response    : raw LLM output
    user_text   : original user message (for signal detection)
    intent      : classified intent (some intents skip steps)
    language    : "de" | "en" — selects correct jargon map and signals
    add_followup: whether to append a follow-up question

    Steps
    -----
    1. Detect confusion / frustration in user_text
    2. Replace jargon
    3. Shorten long sentences
    4. Truncate to max 5 sentences
    5. Optionally add follow-up offer
    """
    
    response = _EMOJI_RE.sub("", response)
    response = re.sub(r"  +", " ", response).strip()

    # ── Confusion prefix ──────────────────────────────────────────────────
    if user_text and is_confused(user_text, language):
        logger.info(f"Confusion detected [{language}] — adding clarification prefix")
        prefix = (
            "Entschuldigung, lassen Sie mich das einfacher erklären. "
            if language == "de" else
            "I'm sorry, let me explain that more simply. "
        )
        response = prefix + response

    # ── Frustration → offer human contact ────────────────────────────────
    if user_text and is_frustrated(user_text, language):
        logger.info(f"Frustration detected [{language}] — offering human contact")
        if language == "en":
            return (
                "I'm sorry you're having difficulties. "
                "I'd be happy to connect you with a real youwel team member "
                "who can help you personally. "
                "Shall I arrange that for you?"
            )
        else:
            return (
                "Es tut mir leid, dass Sie Schwierigkeiten haben. "
                "Ich verbinde Sie gerne mit einem echten youwel-Mitarbeiter, "
                "der Ihnen persönlich helfen kann. "
                "Soll ich das für Sie veranlassen?"
            )

    # ── Emergency — never modify ──────────────────────────────────────────
    if intent == "emergency":
        return response

    # ── Standard pipeline ─────────────────────────────────────────────────
    response = replace_jargon(response, language)
    response = shorten_sentences(response, language=language)

    _list_intents = {"list_capabilities", "youwel_explain", "general_chat"}
    is_numbered_list = bool(re.search(r'^\s*\d+\.', response, re.MULTILINE))
    if not (intent in _list_intents and is_numbered_list):
        response = truncate_response(response, max_sentences=5)

    if add_followup and intent not in ("emergency", "farewell", "thanks"):
        response = add_followup_offer(response, language)

    return response.strip()


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        (
            "Um die DSGVO-Konformität sicherzustellen, werden Ihre Daten über eine "
            "gesicherte API an unsere Server übertragen und im Browser lokal gecacht. "
            "Der Download der Daten erfolgt verschlüsselt via TLS 1.3.",
            "Sind meine Daten sicher?", "youwel_privacy", "de",
        ),
        (
            "To ensure GDPR compliance, your data is transmitted via a secure API "
            "to our servers and cached locally in the browser. "
            "The download is encrypted using TLS 1.3.",
            "Is my data safe?", "youwel_privacy", "en",
        ),
        (
            "Hallo! Schön, dass Sie sich melden. Wie kann ich Ihnen heute helfen?",
            "Hallo Silvie", "greeting", "de",
        ),
        (
            "Hello! Great to hear from you. How can I help you today?",
            "Hello Silvie", "greeting", "en",
        ),
    ]
    for resp, user, intent, lang in test_cases:
        formatted = format_for_senior(resp, user_text=user, intent=intent, language=lang)
        print(f"\nLang    : {lang.upper()}")
        print(f"Original: {resp[:80]}…")
        print(f"Result  : {formatted}")