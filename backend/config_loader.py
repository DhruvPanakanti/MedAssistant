import os
import json

# Anchored to this file's own directory, not the process's current working
# directory, so every path built from it resolves correctly regardless of
# where `python app.py` (or pytest, or train.py) is actually launched from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "diseases.json")


def load_disease_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_disease(disease_key):
    config = load_disease_config()
    if disease_key not in config:
        raise ValueError(f"Unknown condition '{disease_key}'. Available: {list(config.keys())}")
    return config[disease_key]
