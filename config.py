"""Central configuration. Values come from model_config.json (override with env vars)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "model_config.json"

_DEFAULTS = {
    "model_version": "0.0.0",
    "decision_threshold": 0.5,
    "model_path": "models/churn_model.joblib",
}


def load_config():
    cfg = dict(_DEFAULTS)
    if CONFIG_FILE.exists():
        cfg.update(json.loads(CONFIG_FILE.read_text()))
    if os.getenv("CHURN_THRESHOLD"):
        cfg["decision_threshold"] = float(os.environ["CHURN_THRESHOLD"])
    if os.getenv("MODEL_PATH"):
        cfg["model_path"] = os.environ["MODEL_PATH"]
    path = Path(cfg["model_path"])
    cfg["model_path"] = str(path if path.is_absolute() else ROOT / path)
    return cfg
