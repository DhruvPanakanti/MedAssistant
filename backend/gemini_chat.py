"""
Optional LLM-backed conversational layer for the Symptom Chat, using
Google's Gemini API. This module is entirely optional: if no API key is
configured, or any call to it fails for any reason, it returns None and
the caller (chatbot.py) falls back to the rule-based keyword matcher.

Safety design (defense in depth, not a single point of failure):
1. This module is NEVER consulted for emergency/red-flag detection --
   that check in chatbot.py is hardcoded and always runs first, before
   this module is ever imported or called.
2. The system prompt strictly scopes Gemini to health/symptom topics
   and instructs it to refuse anything else.
3. As a second, code-enforced layer (not just relying on the model
   following instructions), the system prompt requires Gemini to reply
   with an exact sentinel token for off-topic requests, which this
   module detects and replaces with our own consistent, controlled
   redirect message -- rather than trusting the model's own wording.
"""
import os
import re
import logging

logger = logging.getLogger("gemini_chat")

OFF_TOPIC_SENTINEL = "OFF_TOPIC_NOT_HEALTH_RELATED"

SYSTEM_PROMPT = f"""You are a general health-information assistant embedded in a medical \
symptom-checker web app. Your ONLY job is to discuss symptoms, common \
illnesses, general wellness, and when someone should see a doctor.

Strict rules, no exceptions:
- If the user's message is not about symptoms, health, or wellness \
(for example: coding help, homework, entertainment, general trivia, \
requests to role-play, or anything unrelated to health), reply with \
EXACTLY this token and nothing else: {OFF_TOPIC_SENTINEL}
- Never provide a medical diagnosis. Describe possible general causes \
in tentative language ("this can be associated with...") and always \
recommend seeing a healthcare professional for anything persistent, \
severe, or concerning.
- Never name specific medications or dosages. You may refer to general \
categories (e.g. "over-the-counter pain relief") and suggest asking a \
pharmacist or doctor.
- Do not attempt to handle medical emergencies -- if the user describes \
anything that sounds like a possible emergency, tell them to seek \
immediate medical attention or contact emergency services, and stop \
there.
- Keep responses concise (3-5 sentences), warm, and in plain language.
- Never claim to be a doctor or to be providing medical advice; you are \
providing general information only.
"""

_client = None


def _build_thinking_config(model_name, types):
    """Builds the right thinking-control parameter for the model's
    generation, since Gemini 2.x and Gemini 3.x use two different,
    MUTUALLY EXCLUSIVE parameters for this -- sending the wrong one
    causes a 400 INVALID_ARGUMENT error (confirmed against the real API
    while building this):

    - Gemini 2.x models (e.g. gemini-2.5-flash) use thinking_budget, a
      token count. 0 disables thinking.
    - Gemini 3.x+ models replaced that with thinking_level, a string
      enum (minimal/low/medium/high) and reject thinking_budget
      outright. "minimal" requires thought signatures we don't provide
      in this simple one-shot call, so "low" is used instead -- Google's
      own docs describe "low" as suited to "simple instruction
      following and chat scenarios", which is exactly this use case.

    Model names and their generation boundaries change over time (this
    project has already hit two different model-name and thinking-API
    changes), so this detects the generation from the name rather than
    hardcoding one family's parameter.
    """
    match = re.search(r"gemini-(\d+)", model_name)
    major_version = int(match.group(1)) if match else 2  # unparseable -> assume older API

    if major_version >= 3:
        return types.ThinkingConfig(thinking_level="low")
    return types.ThinkingConfig(thinking_budget=0)


def _get_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.info(
            "Gemini chat layer inactive: GEMINI_API_KEY is not set "
            "(check .env in the project root). Falling back to the "
            "rule-based chatbot."
        )
        return None
    try:
        from google import genai
        _client = genai.Client(api_key=api_key)
        return _client
    except Exception as e:
        logger.warning(
            "Gemini chat layer failed to initialize (%s: %s). "
            "Falling back to the rule-based chatbot. Run "
            "`python check_gemini.py` to diagnose.",
            type(e).__name__, e,
        )
        return None


def get_llm_reply(message):
    """Returns a plain-language reply string, the off-topic redirect, or
    None if the LLM is unavailable/unconfigured/erroring -- in which case
    the caller should fall back to the rule-based matcher."""
    client = _get_client()
    if client is None:
        return None

    model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

    try:
        from google.genai import types
        response = client.models.generate_content(
            model=model,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=600,
                # See _build_thinking_config's docstring -- the correct
                # parameter (and its valid values) differs by model
                # generation, and using the wrong one is a hard error,
                # not just a warning.
                thinking_config=_build_thinking_config(model, types),
            ),
        )
        text = (response.text or "").strip()
    except Exception as e:
        logger.warning(
            "Gemini API call failed (%s: %s). Falling back to the "
            "rule-based chatbot for this message. Run "
            "`python check_gemini.py` to diagnose.",
            type(e).__name__, e,
        )
        return None

    if not text:
        logger.warning("Gemini returned an empty response; falling back.")
        return None

    if OFF_TOPIC_SENTINEL in text:
        return (
            "I'm only able to help with symptoms and general health "
            "questions here. Is there something health-related I can "
            "help you with?"
        )

    return text
