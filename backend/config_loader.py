import json

CONFIG_PATH = "config/diseases.json"


def load_disease_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_disease(disease_key):
    config = load_disease_config()
    if disease_key not in config:
        raise ValueError(f"Unknown condition '{disease_key}'. Available: {list(config.keys())}")
    return config[disease_key]
