"""Configuration loading. Tunables live in config.yaml, secrets in environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

try:  # .env support is a convenience, not a requirement
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    if load_dotenv is not None:
        load_dotenv(Path(path).resolve().parent / ".env")
        load_dotenv()
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} is empty or invalid")
    return cfg


def env_secret(name: str | None) -> str | None:
    """Read a secret from the environment. Empty strings count as missing."""
    if not name:
        return None
    value = os.environ.get(name, "").strip()
    return value or None
