"""
Loads general, non-personalized lifestyle/dietary guidance per condition.
This is static educational content, not derived from any model or from
a specific user's inputs — it does not vary by prediction result, and
is never treated as personalized medical or dietary advice.
"""
import os
import json

from config_loader import PROJECT_ROOT

TIPS_PATH = os.path.join(PROJECT_ROOT, "data", "lifestyle_tips.json")
_tips_cache = None


def get_lifestyle_tips(disease_key):
    """Returns {"eat_more": [...], "limit": [...], "lifestyle": [...]}
    for a condition, or None if no tips are defined for it."""
    global _tips_cache
    if _tips_cache is None:
        with open(TIPS_PATH) as f:
            _tips_cache = json.load(f)
    return _tips_cache.get(disease_key)
