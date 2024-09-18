"""Safe image decoding, normalisation and image-quality assessment.

All image input in DefectLens (uploads, batch ZIPs, camera frames, datasets) goes through
:func:`decode_image_bytes` / :func:`load_image` so that format whitelisting, decompression-bomb
protection and corrupt-file handling are applied consistently.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_PIXELS = 40_000_000  # ~40 MP; larger images are rejected (decompression-bomb guard)
MIN_SIDE = 32

# PIL raises DecompressionBombError above 2x this value; we additionally check explicitly.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ImageDecodeError(ValueError):
    """Raised when an image cannot be safely decoded. The message is safe to show to users."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def decode_image_bytes(data: bytes, max_pixels: int = MAX_PIXELS) -> np.ndarray:
    """Decode bytes into an RGB uint8 array (H, W, 3).

    Rejects empty input, unsupported formats, truncated/corrupt files, images that are too
    small and images exceeding ``max_pixels``. EXIF orientation is applied.
    """
    if not data:
        raise ImageDecodeError("Empty file")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = (probe.format or "").upper()
            if fmt not in ALLOWED_FORMATS:
                raise ImageDecodeError(f"Unsupported image format '{fmt or 'unknown'}'")
            w, h = probe.size
            if w * h > max_pixels:
                raise ImageDecodeError(f"Image too large ({w}x{h}); limit is {max_pixels} pixels")
            if min(w, h) < MIN_SIDE:
                raise ImageDecodeError(f"Image too small ({w}x{h}); minimum side is {MIN_SIDE}px")
            probe.verify()  # detects many truncated / corrupt files
        with Image.open(io.BytesIO(data)) as raw:
            raw.load()
            im: Image.Image = ImageOps.exif_transpose(raw)
            if im.mode in ("I;16", "I;16B", "I"):
                arr16 = np.asarray(im, dtype=np.float32)
                lo, hi = float(np.min(arr16)), float(np.max(arr16))
                arr8 = ((arr16 - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(arr16)).astype(np.uint8)
                return np.ascontiguousarray(np.stack([arr8] * 3, axis=-1))
            rgb = im.convert("RGB")
            return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    except ImageDecodeError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageDecodeError("Image exceeds the maximum allowed pixel count") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageDecodeError(f"Corrupt or unreadable image: {exc.__class__.__name__}") from exc


def load_image(path: str | Path, max_pixels: int = MAX_PIXELS) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ImageDecodeError(f"Unsupported file extension '{p.suffix}'")
    try:
        data = p.read_bytes()
    except FileNotFoundError as exc:
        raise ImageDecodeError(f"File not found: {p.name}") from exc
    return decode_image_bytes(data, max_pixels=max_pixels)


def load_mask(path: str | Path | None, shape: tuple[int, int]) -> np.ndarray:
    """Load a binary mask (any non-zero pixel = defect). Returns zeros if path is None."""
    if path is None or str(path) == "":
        return np.zeros(shape, dtype=np.uint8)
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    mask: np.ndarray = (arr > 0).astype(np.uint8)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def resize_rgb(img: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    if isinstance(size, int):
        size = (size, size)
    h, w = size
    if img.shape[0] == h and img.shape[1] == w:
        return img
    interp = cv2.INTER_AREA if img.shape[0] > h else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


@dataclass
class QualityReport:
    """Image-quality indicators used by the decision policy to route poor images to REVIEW."""

    width: int
    height: int
    brightness: float  # mean luminance 0..255
    contrast: float  # std of luminance
    sharpness: float  # variance of Laplacian (on a 512px-normalised image)
    saturated_fraction: float  # fraction of clipped (near 0 or 255) pixels
    issues: list[str]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


@dataclass
class QualityLimits:
    min_side: int = 64
    min_brightness: float = 25.0
    max_brightness: float = 235.0
    min_contrast: float = 8.0
    min_sharpness: float = 15.0
    max_saturated_fraction: float = 0.35


def assess_quality(img: np.ndarray, limits: QualityLimits | None = None) -> QualityReport:
    limits = limits or QualityLimits()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    norm = resize_max_side(gray, 512)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(norm, cv2.CV_64F).var())
    saturated = float(((gray <= 2) | (gray >= 253)).mean())
    issues: list[str] = []
    if min(h, w) < limits.min_side:
        issues.append(f"resolution too low ({w}x{h})")
    if brightness < limits.min_brightness:
        issues.append(f"image too dark (mean luminance {brightness:.0f})")
    if brightness > limits.max_brightness:
        issues.append(f"image overexposed (mean luminance {brightness:.0f})")
    if contrast < limits.min_contrast:
        issues.append(f"contrast too low ({contrast:.1f})")
    if sharpness < limits.min_sharpness:
        issues.append(f"image blurry (Laplacian variance {sharpness:.1f})")
    if saturated > limits.max_saturated_fraction:
        issues.append(f"{saturated:.0%} of pixels clipped")
    return QualityReport(w, h, round(brightness, 2), round(contrast, 2), round(sharpness, 2),
                         round(saturated, 4), issues)


def dhash(img: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """Difference hash as a boolean vector (hash_size*hash_size bits) for near-duplicate detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()
