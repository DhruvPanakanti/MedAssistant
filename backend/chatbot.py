"""
Rule-based general symptom checker. Deliberately NOT a diagnostic tool:
it matches free-text symptom descriptions against a small knowledge base
of common, low-severity conditions, flags red-flag/emergency language,
and always recommends professional care for anything serious or
persistent. It never gives specific drug names or dosages.
"""
import os
import json
import re

from config_loader import PROJECT_ROOT

KB_PATH = os.path.join(PROJECT_ROOT, "data", "chatbot_kb.json")

DISCLAIMER = (
    "This is general information, not a medical diagnosis. If you're "
    "ever unsure or symptoms are severe, please contact a doctor or "
    "healthcare provider."
)

GREETINGS = {"hi", "hello", "hey", "hii", "hiya", "yo"}


def _load_kb():
    with open(KB_PATH) as f:
        return json.load(f)


def _normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def check_red_flags(text):
    norm = _normalize(text)
    kb = _load_kb()
    matched = [flag for flag in kb["red_flags"] if flag in norm]
    return matched


def match_conditions(text, top_n=3):
    norm = _normalize(text)
    kb = _load_kb()
    scored = []
    for condition in kb["conditions"]:
        score = sum(1 for kw in condition["keywords"] if kw in norm)
        if score > 0:
            scored.append((score, condition))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_n]]


def generate_reply(message):
    """Returns {type, text, matches} where type is one of:
    'urgent', 'matches', 'no_match', 'greeting'."""
    norm = _normalize(message)

    red_flags = check_red_flags(message)
    if red_flags:
        return {
            "type": "urgent",
            "text": (
                "What you're describing could be a medical emergency. "
                "Please seek immediate medical attention — contact emergency "
                "services or go to the nearest emergency room now. This chat "
                "can't help with an emergency."
            ),
            "matches": []
        }

    if norm in GREETINGS or len(norm) < 2:
        return {
            "type": "greeting",
            "text": (
                "Hi! Tell me what symptoms you're experiencing — for example "
                "\"I have a fever and sore throat\" — and I'll share some "
                "general information. " + DISCLAIMER
            ),
            "matches": []
        }

    matches = match_conditions(message)
    if not matches:
        return {
            "type": "no_match",
            "text": (
                "I don't have general information matching that description. "
                "Try describing your main symptoms more specifically (e.g. "
                "\"headache and nausea\" or \"cough and fever\"), or consult a "
                "healthcare provider for anything persistent or concerning. "
                + DISCLAIMER
            ),
            "matches": []
        }

    lines = []
    for c in matches:
        lines.append(
            f"**{c['name']}**\n{c['advice']}\nSee a doctor if: {c['see_doctor_if']}"
        )
    text = "This could be related to:\n\n" + "\n\n".join(lines) + "\n\n" + DISCLAIMER

    return {"type": "matches", "text": text, "matches": [c["name"] for c in matches]}
