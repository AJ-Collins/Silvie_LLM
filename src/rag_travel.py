"""
src/rag_travel.py
==================
Travel planning using RAG (Retrieval-Augmented Generation).

Uses FREE APIs — no credit card required:
  1. OpenTripMap   — attractions worldwide  (get free key at opentripmap.io)
  2. Wikipedia     — general destination info (no key needed)
  3. Nominatim     — geocoding (OpenStreetMap, no key needed)

Flow:
  user query → extract city/destination → geocode → fetch attractions
  → build context string → inject into LLM prompt → generate response

Set environment variable:
  OPENTRIPMAP_KEY=your_key_here
  (or leave empty to use Wikipedia only)
"""

import os
import re
import json
import logging
import urllib.request
import urllib.parse
from typing import Optional

logger = logging.getLogger(__name__)

OPENTRIPMAP_KEY = os.getenv("OPENTRIPMAP_KEY", "")

# City/destination extraction

# German prepositions used before destinations
_DE_TRAVEL_PATTERNS = [
    r"nach\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)",
    r"in\s+(?:die|den|das|der)?\s*([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)",
    r"(?:nach|in|nach)\s+([A-ZÄÖÜ][a-zäöüß]+)",
    r"([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\s+(?:reisen|besuchen|fahren|fliegen)",
    r"urlaub\s+(?:in|auf)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)",
    r"reise\s+(?:nach|in)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)",
]

def extract_destination(text: str) -> Optional[str]:
    """Extract city/destination name from German text."""
    for pattern in _DE_TRAVEL_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            dest = m.group(1).strip()
            # Filter noise
            if len(dest) > 2 and dest.lower() not in {"ich", "wir", "sie", "er", "es"}:
                return dest
    return None


# Geocoding (Nominatim, no key needed)

def geocode(place: str) -> Optional[tuple[float, float]]:
    """Return (lat, lon) for a place name using OpenStreetMap Nominatim."""
    try:
        q = urllib.parse.quote(place)
        url = (
            f"https://nominatim.openstreetmap.org/search"
            f"?q={q}&format=json&limit=1"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Silvie/3.0 youwel.app"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logger.warning(f"Geocode failed for '{place}': {e}")
    return None


# OpenTripMap attractions

def fetch_opentripmap_attractions(
    lat: float, lon: float, radius_m: int = 3000, limit: int = 8
) -> list[dict]:
    """
    Fetch nearby tourist attractions from OpenTripMap.
    Returns list of {name, kinds, dist} dicts.
    """
    if not OPENTRIPMAP_KEY:
        return []
    try:
        url = (
            f"https://api.opentripmap.com/0.1/en/places/radius"
            f"?radius={radius_m}&lon={lon}&lat={lat}"
            f"&limit={limit}&format=json"
            f"&rate=3&kinds=interesting_places,museums,historic,natural"
            f"&apikey={OPENTRIPMAP_KEY}"
        )
        with urllib.request.urlopen(url, timeout=6) as r:
            data = json.loads(r.read())
        results = []
        for item in data:
            name = item.get("name", "").strip()
            if name:
                results.append({
                    "name": name,
                    "kinds": item.get("kinds", ""),
                    "dist": int(item.get("dist", 0)),
                })
        return results
    except Exception as e:
        logger.warning(f"OpenTripMap failed: {e}")
        return []


# Wikipedia summary

def fetch_wikipedia_summary(place: str, lang: str = "de") -> Optional[str]:
    """
    Fetch a short Wikipedia summary for a place (German by default).
    Returns first ~3 sentences.
    """
    try:
        q = urllib.parse.quote(place.replace(" ", "_"))
        url = (
            f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{q}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Silvie/3.0 youwel.app"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        extract = data.get("extract", "")
        # Return first 3 sentences
        sentences = re.split(r"(?<=[.!?])\s+", extract)
        return " ".join(sentences[:3]) if sentences else None
    except Exception as e:
        logger.warning(f"Wikipedia failed for '{place}': {e}")
        return None


# Main RAG context builder

def build_travel_context(destination: str) -> str:
    """
    Build a context string about a destination using free APIs.
    Returns a string to inject into the LLM prompt.
    """
    context_parts = [f"Reiseziel: {destination}"]

    # 1. Wikipedia summary
    wiki = fetch_wikipedia_summary(destination, lang="de")
    if wiki:
        context_parts.append(f"Allgemeine Information: {wiki}")

    # 2. Geocode + attractions
    coords = geocode(destination)
    if coords:
        lat, lon = coords
        attractions = fetch_opentripmap_attractions(lat, lon)
        if attractions:
            names = [a["name"] for a in attractions[:6]]
            context_parts.append(f"Sehenswürdigkeiten in der Nähe: {', '.join(names)}")

    return "\n".join(context_parts)


# Full RAG pipeline

def get_travel_rag_context(user_message: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract destination from user message and build context.

    Returns
    -------
    (destination, context_string) or (None, None) if no destination found.
    """
    dest = extract_destination(user_message)
    if not dest:
        return None, None

    logger.info(f"RAG: extracting info for destination '{dest}'")
    context = build_travel_context(dest)
    return dest, context


# Travel prompt builder (used by predict_v2.py)

SYSTEM_PROMPT_BASE = """Du bist Silvie, die freundliche KI-Sprachassistentin von youwel.app.
Du hilfst aktiven Menschen ab 40 Jahren in Deutschland.
Antworte immer auf Deutsch, in kurzen, klaren Sätzen (maximal 20 Wörter pro Satz).
Sei freundlich, geduldig und empathisch. Vermeide Fachbegriffe.
Bei medizinischen oder rechtlichen Fragen verweise immer an einen Fachspezialisten.
Bei Notfällen nenne sofort die 112."""


def build_travel_prompt(user_message: str, destination: str, context: str) -> str:
    """Build a complete prompt for the LLM with RAG context injected."""
    system = (
        f"{SYSTEM_PROMPT_BASE}\n\n"
        f"Du hast folgende aktuelle Informationen über {destination}:\n"
        f"{context}\n\n"
        f"Nutze diese Informationen, um dem Nutzer zu helfen. "
        f"Nenne konkrete Sehenswürdigkeiten und gib praktische Tipps."
    )
    return system


# CLI demo
if __name__ == "__main__":
    test_messages = [
        "Ich möchte nach Wien reisen.",
        "Was kann ich in München unternehmen?",
        "Ich plane einen Urlaub auf Mallorca.",
        "Können Sie mir etwas über Rom erzählen?",
        "Ich fahre in die Berge — Garmisch.",
    ]
    for msg in test_messages:
        dest, ctx = get_travel_rag_context(msg)
        if dest:
            print(f"\n✈️  Destination: {dest}")
            print(f"   Context:\n{ctx}")
        else:
            print(f"\n❓  No destination found in: '{msg}'")