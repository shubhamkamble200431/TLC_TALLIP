"""Small filesystem / JSON helpers shared across the pipeline."""
from __future__ import annotations

import io
import json
import os
import tempfile
from typing import Any


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def atomic_write_text(path: str, text: str) -> None:
    """Write `text` to `path` atomically (write tmp + rename)."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    try:
        with io.open(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_json(path: str, obj: Any, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, ensure_ascii=False, default=str))


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_lines(path: str, strip_newline: bool = True) -> list:
    with open(path, encoding="utf-8") as f:
        if strip_newline:
            return [ln.rstrip("\n") for ln in f]
        return list(f)


def write_lines(path: str, lines) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
