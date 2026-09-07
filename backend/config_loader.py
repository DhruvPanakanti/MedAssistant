import os
import json
from dotenv import load_dotenv

# Anchored to this file's own directory, not the process's current working
# directory, so every path built from it resolves correctly regardless of
# where `python app.py` (or pytest, or train.py) is actually launched from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "diseases.json")

# Loads .env (same directory as this file, not the process's cwd -- same
# reasoning as PROJECT_ROOT above) into the environment, once, the first
# time any module imports config_loader. Every other module reads its
# settings via os.environ.get(...), so this one call is enough to make
# .env values available everywhere. If .env doesn't exist, this is a
# harmless no-op -- nothing breaks, os.environ.get() falls back to its
# defaults exactly as it did before .env support existed.
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def load_disease_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_disease(disease_key):
    config = load_disease_config()
    if disease_key not in config:
        raise ValueError(f"Unknown condition '{disease_key}'. Available: {list(config.keys())}")
    return config[disease_key]
