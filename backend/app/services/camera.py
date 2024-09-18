"""Server-side camera adapters (USB / IP stream / HTTP snapshot / folder simulator).

Browser webcams are handled entirely in the frontend (getUserMedia). These adapters let the
server grab frames from cameras attached to (or reachable from) the inspection server.
Physical USB/IP cameras could not be validated in CI; the ``folder`` adapter is used as an
automated substitute and is clearly labelled as simulated.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import httpx

from defectml.imaging import ALLOWED_EXTENSIONS

KINDS = ("usb", "ip", "http_snapshot", "folder")
_folder_pos: dict[str, int] = {}
_lock = threading.Lock()


class CameraError(RuntimeError):
    pass


def snapshot(cam: dict, timeout: float = 8.0) -> bytes:
    kind = cam.get("kind")
    src = str(cam.get("source", "")).strip()
    if kind == "folder":
        d = Path(src)
        files = sorted(p for p in d.glob("*") if p.suffix.lower() in ALLOWED_EXTENSIONS) if d.is_dir() else []
        if not files:
            raise CameraError(f"Simulated camera folder '{src}' has no images")
        with _lock:
            i = _folder_pos.get(cam["id"], 0) % len(files)
            _folder_pos[cam["id"]] = i + 1
        return files[i].read_bytes()
    if kind == "http_snapshot":
        try:
            r = httpx.get(src, timeout=timeout)
        except httpx.HTTPError as exc:
            raise CameraError(f"Camera unreachable: {exc.__class__.__name__}") from exc
        if r.status_code != 200 or not r.content:
            raise CameraError(f"Camera returned HTTP {r.status_code}")
        return r.content
    if kind in ("usb", "ip"):
        source: int | str = int(src) if kind == "usb" and src.isdigit() else src
        cap = cv2.VideoCapture(source)
        try:
            if not cap.isOpened():
                raise CameraError("Could not open camera (check index/URL, permissions and cabling)")
            ok, frame = False, None
            for _ in range(3):  # skip stale buffered frames
                ok, frame = cap.read()
            if not ok or frame is None:
                raise CameraError("Camera opened but no frame was received")
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise CameraError("Frame encoding failed")
            return buf.tobytes()
        finally:
            cap.release()
    raise CameraError(f"Unsupported camera kind '{kind}'")
