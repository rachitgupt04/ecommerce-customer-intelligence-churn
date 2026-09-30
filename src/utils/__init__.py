"""
Utils package initialization.
"""

from src.utils.config import *
from src.utils.logger import get_logger
import json
import joblib
from typing import Any, Dict
from pathlib import Path

def save_model(model: Any, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath: Path) -> Any:
    if not filepath.exists():
        raise FileNotFoundError(f"Artifact not found at {filepath}")
    return joblib.load(filepath)

def save_json(data: Dict, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_json(filepath: Path) -> Dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

