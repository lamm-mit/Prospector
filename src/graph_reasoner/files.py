"""Writes that leave no half-written file.

The text goes to `<name>.tmp` beside the target and is renamed over it once
complete, so a reader sees either the old file or the whole new one. The
recipe's plan takes a file's existence as a stage's completion (`recipe/plan.py`),
so a builder or a read interrupted mid-write must leave nothing under the
target's name.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def write_text_atomic(path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl_atomic(path, rows) -> None:
    """One JSON object per line, unicode kept, through the temporary name."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


__all__ = ["write_jsonl_atomic", "write_text_atomic"]
