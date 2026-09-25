"""Atomic and deterministic filesystem helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch
import yaml


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and return its absolute path."""
    result = Path(path).expanduser().resolve()
    result.mkdir(parents=True, exist_ok=True)
    return result


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Serialize JSON through an atomic same-directory replacement."""
    target = Path(path)

    def writer(temporary: Path) -> None:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")

    _atomic_write(target, writer)


def atomic_write_yaml(path: str | Path, value: Any) -> None:
    """Serialize YAML through an atomic same-directory replacement."""
    target = Path(path)

    def writer(temporary: Path) -> None:
        temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    _atomic_write(target, writer)


def atomic_torch_save(path: str | Path, value: Any) -> None:
    target = Path(path)
    _atomic_write(target, lambda temporary: torch.save(value, temporary))


def atomic_write_csv(path: str | Path, frame: Any) -> None:
    """Commit an entire metrics history before publishing its checkpoint."""
    _atomic_write(Path(path), lambda temporary: frame.to_csv(temporary, index=False))


def config_fingerprint(value: Any) -> str:
    """Create a stable SHA-256 fingerprint for a JSON-compatible configuration."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: str | Path) -> Any:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {target}")
    return json.loads(target.read_text(encoding="utf-8"))
