"""Structured logging (JSON or text)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

_EXTRA = ("request_id", "status", "duration_ms", "job_id", "inspection_id", "path")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": datetime.now(UTC).isoformat(), "level": record.levelname, "logger": record.name,
             "msg": record.getMessage()}
        for k in _EXTRA:
            if hasattr(record, k):
                d[k] = getattr(record, k)
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, default=str)


def setup_logging(fmt: str = "text", level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter() if fmt == "json" else
                   logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(h)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)
