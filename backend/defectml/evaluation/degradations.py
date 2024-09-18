"""Realistic, severity-controlled image degradations for robustness testing.

Each function takes (img uint8 RGB, mask uint8 or None, severity 1..5, rng) and returns
(img, mask). Geometric transforms apply the same warp to the mask so localisation metrics stay valid.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

Degr = Callable[[np.ndarray, np.ndarray | None, int, np.random.Generator], tuple[np.ndarray, np.ndarray | None]]


def _lvl(sev: int, values: list[float]) -> float:
    return values[max(1, min(5, sev)) - 1]


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def brightness(img, mask, sev, rng):
    d = _lvl(sev, [0.1, 0.2, 0.3, 0.4, 0.5]) * (1 if rng.random() < 0.5 else -1)
    return _clip(img.astype(np.float32) * (1 + d)), mask


def contrast(img, mask, sev, rng):
    c = _lvl(sev, [0.8, 0.65, 0.5, 0.4, 0.3])
    f = img.astype(np.float32)
    m = f.mean()
    return _clip((f - m) * c + m), mask


def uneven_lighting(img, mask, sev, rng):
    h, w = img.shape[:2]
    strength = _lvl(sev, [0.15, 0.25, 0.35, 0.45, 0.6])
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - cx, yy - cy) / np.hypot(w, h)
    gain = 1 + strength * (0.5 - d) * 2
    return _clip(img.astype(np.float32) * gain[..., None]), mask


def gaussian_noise(img, mask, sev, rng):
    s = _lvl(sev, [4, 8, 12, 18, 25])
    return _clip(img.astype(np.float32) + rng.normal(0, s, img.shape)), mask


def shot_noise(img, mask, sev, rng):
    lam = _lvl(sev, [60, 30, 15, 8, 4])
    return _clip(rng.poisson(img.astype(np.float32) / 255 * lam) / lam * 255), mask


def sensor_noise(img, mask, sev, rng):
    """Row/column fixed-pattern noise + salt-and-pepper defective pixels."""
    f = img.astype(np.float32)
    s = _lvl(sev, [2, 4, 6, 8, 10])
    f += rng.normal(0, s, (f.shape[0], 1, 1)) + rng.normal(0, s, (1, f.shape[1], 1))
    p = _lvl(sev, [0.001, 0.002, 0.004, 0.007, 0.01])
    sp = rng.random(f.shape[:2])
    f[sp < p / 2] = 0
    f[sp > 1 - p / 2] = 255
    return _clip(f), mask


def defocus_blur(img, mask, sev, rng):
    s = _lvl(sev, [0.6, 1.0, 1.6, 2.3, 3.0]) * max(img.shape[:2]) / 256
    return cv2.GaussianBlur(img, (0, 0), s), mask


def motion_blur(img, mask, sev, rng):
    k = int(_lvl(sev, [3, 5, 9, 13, 17]) * max(img.shape[:2]) / 256) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1 / k
    ang = rng.uniform(0, 180)
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel), mask


def _warp(img, mask, m, perspective=False):
    h, w = img.shape[:2]
    if perspective:
        out = cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        om = cv2.warpPerspective(mask, m, (w, h), flags=cv2.INTER_NEAREST) if mask is not None else None
    else:
        out = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        om = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST) if mask is not None else None
    return out, om


def rotation(img, mask, sev, rng):
    a = _lvl(sev, [2, 4, 8, 12, 20]) * (1 if rng.random() < 0.5 else -1)
    h, w = img.shape[:2]
    return _warp(img, mask, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0))


def translation(img, mask, sev, rng):
    h, w = img.shape[:2]
    f = _lvl(sev, [0.02, 0.04, 0.06, 0.09, 0.12])
    m = np.float32([[1, 0, f * w * rng.choice([-1, 1])], [0, 1, f * h * rng.choice([-1, 1])]])
    return _warp(img, mask, m)


def perspective(img, mask, sev, rng):
    h, w = img.shape[:2]
    f = _lvl(sev, [0.02, 0.04, 0.06, 0.08, 0.11])
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-f, f, (4, 2)).astype(np.float32) * np.float32([w, h])
    return _warp(img, mask, cv2.getPerspectiveTransform(src, dst), perspective=True)


def orientation_flip(img, mask, sev, rng):
    """Part presented mirrored / rotated 180 deg (severity >=3 rotates by 180)."""
    if sev >= 3:
        return np.ascontiguousarray(img[::-1, ::-1]), (np.ascontiguousarray(mask[::-1, ::-1]) if mask is not None else None)
    return np.ascontiguousarray(img[:, ::-1]), (np.ascontiguousarray(mask[:, ::-1]) if mask is not None else None)


def jpeg_compression(img, mask, sev, rng):
    q = int(_lvl(sev, [60, 40, 25, 15, 8]))
    _ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), mask


def resolution_loss(img, mask, sev, rng):
    f = _lvl(sev, [0.75, 0.5, 0.35, 0.25, 0.18])
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), mask


def color_temperature(img, mask, sev, rng):
    t = _lvl(sev, [0.05, 0.1, 0.15, 0.22, 0.3]) * (1 if rng.random() < 0.5 else -1)
    gains = np.array([1 + t, 1.0, 1 - t], np.float32)  # warm (+) or cool (-)
    return _clip(img.astype(np.float32) * gains), mask


def white_balance(img, mask, sev, rng):
    s = _lvl(sev, [0.04, 0.08, 0.12, 0.17, 0.22])
    gains = (1 + rng.uniform(-s, s, 3)).astype(np.float32)
    return _clip(img.astype(np.float32) * gains), mask


def camera_quality(img, mask, sev, rng):
    """Low-cost camera: downsample + blur + noise + JPEG."""
    img, mask = resolution_loss(img, mask, max(1, sev - 1), rng)
    img, mask = gaussian_noise(img, mask, max(1, sev - 1), rng)
    return jpeg_compression(img, mask, sev, rng)


def occlusion(img, mask, sev, rng):
    """Mild occlusion (e.g. tool / finger / tape) by a grey rectangle; covered defect pixels are removed."""
    h, w = img.shape[:2]
    f = _lvl(sev, [0.04, 0.06, 0.08, 0.11, 0.14])
    ow, oh = int(w * f * 2), int(h * f)
    x, y = int(rng.uniform(0, w - ow)), int(rng.uniform(0, h - oh))
    out = img.copy()
    out[y:y + oh, x:x + ow] = rng.integers(60, 200)
    om = mask
    if mask is not None:
        om = mask.copy()
        om[y:y + oh, x:x + ow] = 0
    return out, om


def background_change(img, mask, sev, rng):
    """Shift colour of dark background pixels (e.g. new conveyor belt / fixture colour)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bg = gray < np.percentile(gray, 15)
    shift = np.array(rng.uniform(-1, 1, 3) * _lvl(sev, [10, 20, 30, 45, 60]), np.float32)
    f = img.astype(np.float32)
    f[bg] += shift
    return _clip(f), mask


DEGRADATIONS: dict[str, Degr] = {
    "brightness": brightness, "contrast": contrast, "uneven_lighting": uneven_lighting,
    "gaussian_noise": gaussian_noise, "shot_noise": shot_noise, "sensor_noise": sensor_noise,
    "defocus_blur": defocus_blur, "motion_blur": motion_blur,
    "rotation": rotation, "translation": translation, "perspective": perspective, "orientation": orientation_flip,
    "jpeg": jpeg_compression, "resolution_loss": resolution_loss,
    "color_temperature": color_temperature, "white_balance": white_balance,
    "camera_quality": camera_quality, "occlusion": occlusion, "background_change": background_change,
}

GROUPS = {
    "lighting": ["brightness", "contrast", "uneven_lighting"],
    "noise": ["gaussian_noise", "shot_noise", "sensor_noise"],
    "blur": ["defocus_blur", "motion_blur"],
    "geometry": ["rotation", "translation", "perspective", "orientation"],
    "compression": ["jpeg", "resolution_loss"],
    "color": ["color_temperature", "white_balance"],
    "camera": ["camera_quality"],
    "scene": ["occlusion", "background_change"],
}


def apply(name: str, img: np.ndarray, mask: np.ndarray | None, severity: int, seed: int = 0):
    if name not in DEGRADATIONS:
        raise ValueError(f"Unknown degradation '{name}'")
    rng = np.random.default_rng(seed)
    return DEGRADATIONS[name](img, mask, severity, rng)
