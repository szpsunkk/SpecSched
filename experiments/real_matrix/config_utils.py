"""Shared helpers for configured real-matrix experiments."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import NamedTuple


class LoadedConfig(NamedTuple):
    cfg: dict
    raw_text: str


def validate_schema(cfg: dict) -> None:
    version = cfg.get("schema_version")
    if version != 1:
        raise ValueError(f"unsupported config schema_version={version!r}; expected 1")


def expand_seed_placeholders(value, seed: int):
    if isinstance(value, str):
        return value.replace("{seed}", str(seed))
    if isinstance(value, list):
        return [expand_seed_placeholders(item, seed) for item in value]
    if isinstance(value, dict):
        return {key: expand_seed_placeholders(item, seed) for key, item in value.items()}
    return value


def load_config(path: Path, root: Path) -> LoadedConfig:
    raw_text = path.read_text(encoding="utf-8")
    os.environ.setdefault("PROJECT_ROOT", str(root))
    cfg = json.loads(os.path.expandvars(raw_text))
    validate_schema(cfg)
    seed = int(cfg.get("experiment", {}).get("seed", 0))
    cfg = expand_seed_placeholders(cfg, seed)
    return LoadedConfig(cfg=cfg, raw_text=raw_text)


def config_file_hash(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:12]


def runtime_config_hash(cfg: dict) -> str:
    """Fingerprint of the expanded config actually used on this machine."""
    payload = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def hyperparam_hash(cfg: dict) -> str:
    """Path-independent fingerprint for comparing experiment hyperparameters."""
    stripped = {
        key: value
        for key, value in cfg.items()
        if key not in ("paths", "models")
    }
    payload = json.dumps(stripped, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]
