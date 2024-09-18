"""Append-only, hash-chained audit trail."""

from __future__ import annotations

import hashlib
import json
import threading

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AuditEvent, User, utcnow

_lock = threading.Lock()


def _digest(prev: str, ev: AuditEvent) -> str:
    payload = json.dumps({"prev": prev, "ts": ev.created_at.isoformat(), "actor": ev.actor_id, "action": ev.action,
                          "type": ev.entity_type, "id": ev.entity_id, "details": ev.details},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def record(db: Session, action: str, actor: User | None = None, entity_type: str = "", entity_id: str | int = "",
           details: dict | None = None) -> AuditEvent:
    """Add an audit event to the current transaction (committed by the caller)."""
    with _lock:
        last = db.execute(select(AuditEvent.hash).order_by(AuditEvent.id.desc()).limit(1)).scalar()
        ev = AuditEvent(created_at=utcnow(), actor_id=actor.id if actor else None,
                        actor_name=actor.username if actor else "system", action=action, entity_type=entity_type,
                        entity_id=str(entity_id), details=details or {}, prev_hash=last or "")
        ev.hash = _digest(ev.prev_hash, ev)
        db.add(ev)
        db.flush()
    return ev


def verify_chain(db: Session) -> dict:
    prev = ""
    n = 0
    for ev in db.execute(select(AuditEvent).order_by(AuditEvent.id)).scalars():
        created = ev.created_at
        if created.tzinfo is None:  # SQLite drops tz info; hash was computed with UTC-aware timestamps
            from datetime import UTC

            ev_created = created.replace(tzinfo=UTC)
        else:
            ev_created = created
        probe = AuditEvent(created_at=ev_created, actor_id=ev.actor_id, action=ev.action, entity_type=ev.entity_type,
                           entity_id=ev.entity_id, details=ev.details)
        if ev.prev_hash != prev or _digest(prev, probe) != ev.hash:
            return {"valid": False, "broken_at": ev.id, "checked": n}
        prev = ev.hash
        n += 1
    return {"valid": True, "checked": n}
