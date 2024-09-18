"""Typed key/value application settings stored in the database, with defaults."""

from __future__ import annotations

import copy

from sqlalchemy.orm import Session

from app.models import AppSetting, User, utcnow
from defectml.decision import PolicyConfig
from defectml.imaging import QualityLimits

DEFAULTS: dict[str, dict] = {
    "policy": PolicyConfig().to_dict(),
    "quality": QualityLimits().__dict__,
    "camera": {
        "default_interval_s": 5,
        "default_product": "",
        "mirror_preview": False,
        "resolution": "1280x720",
        "cameras": [],  # [{id, name, kind: usb|ip|file, source, enabled}]
    },
    "privacy": {
        "allow_external_llm": False,  # master switch for any external AI provider
        "include_filenames_in_llm_context": False,
    },
    "retention": {
        "image_days": 180,  # delete stored images after N days (metadata kept); 0 = keep forever
        "inspection_days": 0,  # delete inspection records after N days; 0 = keep forever
        "job_days": 30,
        "keep_reviewed_images": True,
    },
    "llm": {"active_provider_id": None},
    "general": {"site_name": "DefectLens", "duplicate_window_hours": 24, "operating_mode": "decision_support"},
}


def get_setting(db: Session, key: str) -> dict:
    if key not in DEFAULTS:
        raise KeyError(key)
    row = db.get(AppSetting, key)
    val = copy.deepcopy(DEFAULTS[key])
    if row is not None and isinstance(row.value, dict):
        val.update(row.value)
    return val


def all_settings(db: Session) -> dict:
    return {k: get_setting(db, k) for k in DEFAULTS}


def set_setting(db: Session, key: str, value: dict, user: User | None) -> dict:
    if key not in DEFAULTS:
        raise KeyError(key)
    merged = get_setting(db, key)
    for k, v in value.items():
        if k in DEFAULTS[key]:
            merged[k] = v
    row = db.get(AppSetting, key)
    if row is None:
        row = AppSetting(key=key, value=merged)
        db.add(row)
    else:
        row.value = merged
    row.updated_at = utcnow()
    row.updated_by_id = user.id if user else None
    return merged
