"""Simple in-memory sliding-window rate limiter (per client IP), plus request-ID/log middleware.

For multi-instance deployments put a reverse proxy rate limiter (nginx/traefik) in front.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.metrics import METRICS

log = logging.getLogger("defectlens.http")


def _csp() -> str:
    """Strict same-origin policy: no inline scripts, no third-party origins. Webcam frames use blob:/data: URLs."""
    from app.config import get_settings

    s = get_settings()
    parts = ["default-src 'self'", "script-src 'self'", "style-src 'self' 'unsafe-inline'",
             "img-src 'self' data: blob:", "media-src 'self' blob:", "font-src 'self' data:", "connect-src 'self'",
             "object-src 'none'", "base-uri 'self'", "form-action 'self'"]
    fa = s.frame_ancestors or ("'self'" if s.is_production else "")
    if fa:
        parts.append(f"frame-ancestors {fa}")
    return "; ".join(parts)


class SlidingWindow:
    def __init__(self):
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float = 60.0) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= now - window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


LIMITER = SlidingWindow()


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


class RequestMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit_per_min: int):
        super().__init__(app)
        self.limit = limit_per_min

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        path = request.url.path
        if (path.startswith("/api") and not path.startswith("/api/health") and self.limit > 0
                and not LIMITER.allow(f"all:{client_ip(request)}", self.limit)):
            METRICS.inc("http_rate_limited_total")
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429, headers={"Retry-After": "60"})
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"request_id": rid, "path": path})
            METRICS.inc("http_requests_total", status="500")
            return JSONResponse({"detail": "Internal server error", "request_id": rid}, status_code=500)
        dt = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = rid
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if not path.startswith(("/docs", "/redoc")):  # Swagger/ReDoc pages load their assets from a CDN
            response.headers.setdefault("Content-Security-Policy", _csp())
        response.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(), geolocation=(), payment=()")
        if path.startswith("/api"):
            METRICS.inc("http_requests_total", status=str(response.status_code))
            METRICS.observe("http_request_ms", dt)
            if not path.startswith("/api/health"):
                log.info("%s %s %s %.1fms", request.method, path, response.status_code, dt,
                         extra={"request_id": rid, "status": response.status_code, "duration_ms": round(dt, 1)})
        return response
