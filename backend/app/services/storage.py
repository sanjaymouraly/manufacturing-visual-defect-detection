"""File storage for inspection images and derived artefacts (paths stored relative to data_dir)."""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from app.config import get_settings

SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str, max_len: int = 120) -> str:
    base = os.path.basename(name.replace("\\", "/")) or "file"
    base = SAFE.sub("_", base).strip("._") or "file"
    return base[-max_len:]


def inspection_dir(inspection_id: str, when: datetime) -> tuple[Path, str]:
    rel = f"inspections/{when:%Y/%m/%d}/{inspection_id}"
    d = get_settings().data_dir / rel
    d.mkdir(parents=True, exist_ok=True)
    return d, rel


def abs_data_path(rel: str) -> Path:
    base = get_settings().data_dir.resolve()
    p = (base / rel).resolve()
    if base not in p.parents:
        raise ValueError("Path escapes data directory")
    return p


def delete_inspection_files(rel_dir: str) -> None:
    if not rel_dir:
        return
    p = abs_data_path(rel_dir)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
