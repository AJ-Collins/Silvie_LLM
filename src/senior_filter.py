"""
src/senior_filter.py
=====================
Post-process LLM responses to be senior-friendly:
  - Shorten overly long answers
  - Remove jargon
  - Add confirmation phrasing
  - Enforce sentence length limits
  - Detect and handle confusion/frustration signals
"""

import re
import logging

logger = logging.getLogger(__name__)

# Max tokens in a single response sentence for senior users
MAX_SENTENCE_WORDS = 22

# Jargon replacements (technical → plain German)
_JARGON_MAP: dict[str, str] = {
    "KI":           "Künstliche Intelligenz",
    "AI":           "Künstliche Intelligenz",
    "API":          "unsere digitale Verbindung",
    "DSGVO":        "deutsches Datenschutzgesetz",
    "QR-Code":      "QR-Code (das quadratische Bild zum Scannen)",
    "Browser":      "Internet-Programm",
    "App":          "Anwendung",
    "Upload":       "Hochladen",
    "Download":     "Herunterladen",
    "E-Mail":       "E-Mail",       # already simple, keep
    "SMS":          "Textnachricht",
    "Link":         "Internetadresse",
    "Server":       "Computer-Server",
    "Account":      "Benutzerkonto",
    "Password":     "Passwort",
    "Login":        "Anmeldung",
    "Update":       "Aktualisierung",
}

# Confusion/frustration signals in user messages
_CONFUSION_SIGNALS = {
    "verstehe nicht", "was meinen sie", "ich bin verwirrt",
    "das ergibt keinen sinn", "zu kompliziert", "nicht verstanden",
    "bitte wiederholen", "zu schnell", "was haben sie gesagt",
    "ich komme nicht weiter", "ich weiß nicht was",
}

# Frustration/distress signals
_DISTRESS_SIGNALS = {
    "das ist frustrierend", "ich bin genervt", "das klappt nicht",
    "nichts funktioniert", "ich verstehe das nicht mehr",
    "ich brauche einen menschen", "echten mitarbeiter",
    "ich will mit jemandem sprechen",
}


def is_confused(user_text: str) -> bool:
    t = user_text.lower()
    return any(s in t for s in _CONFUSION_SIGNALS)


def is_frustrated(user_text: str) -> bool:
    t = user_text.lower()
    return any(s in t for s in _DISTRESS_SIGNALS)


def replace_jargon(text: str) -> str:
    """Replace technical terms with plain German equivalents."""
    for jargon, replacement in _JARGON_MAP.items():
        # Case-insensitive whole-word replacement
        text = re.sub(
            rf"\b{re.escape(jargon)}\b", replacement, text, flags=re.IGNORECASE
        )
    return text


def shorten_sentences(text: str, max_words: int = MAX_SENTENCE_WORDS) -> str:
    """
    Break overly long sentences at natural conjunctions.
    Does NOT truncate — splits at 'und', 'aber', 'oder', 'damit', 'weil'.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    result = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            result.append(sentence)
        else:
            # Try to split at a conjunction
            split_words = ["und", "aber", "oder", "damit", "weil", "denn", "jedoch"]
            split_idx = None
            for i, w in enumerate(words[8:max_words], start=8):
                if w.lower() in split_words:
                    split_idx = i
                    break
            if split_idx:
                part1 = " ".join(words[:split_idx])
                part2 = " ".join(words[split_idx:])
                # Capitalize second part
                part2 = part2[0].upper() + part2[1:]
                result.append(part1 + ".")
                result.append(part2)
            else:
                # Can't split nicely — truncate at max_words
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


def add_followup_offer(text: str) -> str:
    """Add a gentle follow-up offer if not already present."""
    follow_up_phrases = [
        "Kann ich", "Haben Sie noch", "Möchten Sie", "Soll ich",
        "Darf ich", "Wünschen Sie",
    ]
    if any(text.rstrip().endswith("?") for _ in [1]):
        return text  # already ends with a question
    if not any(p in text for p in follow_up_phrases):
        text = text.rstrip(".!") + ". Haben Sie noch weitere Fragen?"
    return text


def format_for_senior(
    response: str,
    user_text: str = "",
    intent: str = "",
    add_followup: bool = True,
) -> str:
    """
    Full senior-friendly formatting pipeline.

    Steps:
      1. Replace jargon
      2. Shorten long sentences
      3. Truncate to max 5 sentences
      4. Optionally add follow-up offer
      5. Handle confusion/frustration specially
    """
    # Special handling for confusion
    if user_text and is_confused(user_text):
        logger.info("Confusion detected — adding clarification offer")
        response = (
            "Entschuldigung, lassen Sie mich das einfacher erklären. "
            + response
        )

    # Special handling for frustration → offer human contact
    if user_text and is_frustrated(user_text):
        logger.info("Frustration detected — offering human contact")
        return (
            "Es tut mir leid, dass Sie Schwierigkeiten haben. "
            "Ich verbinde Sie gerne mit einem echten youwel-Mitarbeiter, "
            "der Ihnen persönlich helfen kann. "
            "Soll ich das für Sie veranlassen?"
        )

    # Emergency — never modify
    if intent == "emergency":
        return response

    response = replace_jargon(response)
    response = shorten_sentences(response)
    response = truncate_response(response, max_sentences=5)

    if add_followup and intent not in ("emergency", "farewell", "thanks"):
        response = add_followup_offer(response)

    return response.strip()


#  Demo
if __name__ == "__main__":
    test_responses = [
        (
            "Um die DSGVO-Konformität sicherzustellen, werden Ihre Daten über eine "
            "gesicherte API an unsere Server übertragen und im Browser lokal gecacht. "
            "Der Download der Daten erfolgt verschlüsselt via TLS 1.3.",
            "Sind meine Daten sicher?",
            "youwel_privacy",
        ),
        (
            "Hallo! Schön, dass Sie sich melden. Wie kann ich Ihnen heute helfen?",
            "Hallo Silvie",
            "greeting",
        ),
    ]
    for resp, user, intent in test_responses:
        formatted = format_for_senior(resp, user_text=user, intent=intent)
        print(f"\nOriginal : {resp[:80]}…")
        print(f"Formatted: {formatted[:120]}…")