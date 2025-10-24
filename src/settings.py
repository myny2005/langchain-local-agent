from __future__ import annotations

import json
import os
import pathlib
import hashlib
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG_ENV = "LC_CONFIG"
DEFAULT_CONFIG_PATH = "configs/config.yaml"


class ConfigError(RuntimeError):
    """Raised when configuration files are missing or invalid."""


def _resolve_config_path(path: Optional[str] = None) -> pathlib.Path:
    cand = path or os.getenv(DEFAULT_CONFIG_ENV) or DEFAULT_CONFIG_PATH
    p = pathlib.Path(cand).expanduser().resolve()
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    if not p.is_file():
        raise ConfigError(f"Config path is not a file: {p}")
    return p


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = _resolve_config_path(path)
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in config {cfg_path}: {e}") from e
    if not isinstance(cfg, dict):
        raise ConfigError(
            f"Expected a mapping at the root of {cfg_path}, got {type(cfg)}"
        )
    return cfg


def persist_path_for_url(base_dir: str, url: str, create: bool = True) -> str:
    """
    Generate a stable hashed directory path for a URL under base_dir.
    Optionally creates the directory.

    Example:
        persist_path_for_url('chroma_db', 'https://example.com/article')
        -> '.../chroma_db/1a2b3c4d5e6f7g8h'
    """
    if not base_dir:
        raise ValueError("base_dir must be a non-empty string")
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    path = pathlib.Path(base_dir).expanduser().resolve() / h
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return str(path)


def load_cookies(path: Optional[str]) -> Optional[dict]:
    """
    Load cookies JSON if a path is provided; return None otherwise.
    Accepts either a list of cookie dicts (browser export) or a mapping; returns as-is.
    """
    if not path:
        return None
    p = pathlib.Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Cookies file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Cookies file is not valid JSON: {p}") from e
    if not isinstance(data, (list, dict)):
        raise TypeError(f"Cookies must be a list or dict, got {type(data)} in {p}")
    return data  # type: ignore[return-value]
