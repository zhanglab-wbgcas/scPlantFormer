from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw


def load_model_overrides_from_env() -> Dict[str, Any]:
    model_type = (_env_str("SCPLANT_MODEL_TYPE", "gpt-nano") or "gpt-nano").strip()
    act = (_env_str("SCPLANT_ACT", "relu") or "relu").strip().lower()
    n_layer = _env_int("SCPLANT_N_LAYER")
    n_head = _env_int("SCPLANT_N_HEAD")
    n_embd = _env_int("SCPLANT_N_EMBD")

    if any(v is not None for v in (n_layer, n_head, n_embd)):
        if not all(v is not None for v in (n_layer, n_head, n_embd)):
            raise ValueError("SCPLANT_N_LAYER, SCPLANT_N_HEAD, SCPLANT_N_EMBD must be set together.")
        model_type = None

    return {
        "model_type": model_type,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "act": act,
    }


def load_model_config_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model_config_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
