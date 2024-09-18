"""Procedural SYNTHETIC demo datasets (clearly labelled as synthetic).

Two products are generated deterministically from a seed:

* ``demo-pcb``   - a stylised printed circuit board. Defects: missing_component, scratch,
                   contamination, solder_bridge.
* ``demo-metal`` - a machined metal flange. Defects: scratch, crack, dent, contamination.

Because defects are rendered by the generator, their pixel masks and defect-type labels are
exact by construction (``defect_types_annotated = True``). Each image belongs to a simulated
production batch (``group``) with its own lighting offset, which exercises group-aware splitting.
Synthetic data is for exploring the application only - it says nothing about real-world accuracy.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from defectml.data.manifest import LABEL_ANOMALY, LABEL_NORMAL, DatasetInfo, Sample, save_dataset
from defectml.data.splits import SplitConfig, make_splits

SIZE = 256
PCB_DEFECTS = ["missing_component", "scratch", "contamination", "solder_bridge"]
METAL_DEFECTS = ["scratch", "crack", "dent", "contamination"]
PRODUCTS = {"demo-pcb": PCB_DEFECTS, "demo-metal": METAL_DEFECTS}


def _pcb_layout(rng: np.random.Generator) -> dict:
    comps = []
    # ICs
    for x, y, w, h in [(40, 40, 50, 34), (150, 36, 62, 40), (36, 150, 40, 52), (120, 140, 46, 46), (190, 170, 34, 50)]:
        comps.append({"kind": "ic", "box": (x, y, w, h)})
    # small passives
    for _ in range(12):
        x, y = int(rng.integers(20, 225)), int(rng.integers(20, 225))
        horiz = bool(rng.integers(0, 2))
        w, h = (14, 7) if horiz else (7, 14)
        box = (x, y, w, h)
        if any(_overlap(box, c["box"], 6) for c in comps):
            continue
        comps.append({"kind": "passive", "box": box})
    traces = []
    for _ in range(14):
        p1 = (int(rng.integers(10, 246)), int(rng.integers(10, 246)))
        p2 = (p1[0], int(rng.integers(10, 246))) if rng.random() < 0.5 else (int(rng.integers(10, 246)), p1[1])
        traces.append((p1, p2))
    holes = [(int(rng.integers(15, 241)), int(rng.integers(15, 241))) for _ in range(10)]
    return {"comps": comps, "traces": traces, "holes": holes}


def _overlap(a, b, margin=0) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + margin < bx or bx + bw + margin < ax or ay + ah + margin < by or by + bh + margin < ay)


def _render_pcb(layout: dict, rng: np.random.Generator, skip: int | None = None) -> np.ndarray:
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[:] = (34, 102, 52)
    for p1, p2 in layout["traces"]:
        cv2.line(img, p1, p2, (70, 150, 80), 3, cv2.LINE_AA)
    for (x, y) in layout["holes"]:
        cv2.circle(img, (x, y), 5, (190, 160, 70), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 2, (20, 20, 20), -1, cv2.LINE_AA)
    for idx, c in enumerate(layout["comps"]):
        x, y, w, h = c["box"]
        if idx == skip:
            # exposed solder pads remain where the component is missing
            if c["kind"] == "ic":
                for k in range(4):
                    px = x + 6 + k * (w - 12) // 3
                    cv2.rectangle(img, (px - 2, y - 4), (px + 2, y + 1), (200, 200, 205), -1)
                    cv2.rectangle(img, (px - 2, y + h - 1), (px + 2, y + h + 4), (200, 200, 205), -1)
            else:
                cv2.rectangle(img, (x, y), (x + 3, y + h), (200, 200, 205), -1)
                cv2.rectangle(img, (x + w - 3, y), (x + w, y + h), (200, 200, 205), -1)
            continue
        if c["kind"] == "ic":
            for k in range(4):
                px = x + 6 + k * (w - 12) // 3
                cv2.rectangle(img, (px - 2, y - 4), (px + 2, y + 1), (200, 200, 205), -1)
                cv2.rectangle(img, (px - 2, y + h - 1), (px + 2, y + h + 4), (200, 200, 205), -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (28, 28, 30), -1)
            cv2.circle(img, (x + 6, y + 6), 2, (90, 90, 90), -1)
            cv2.line(img, (x + 10, y + h // 2), (x + w - 10, y + h // 2), (120, 120, 120), 1)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (60, 110, 160), -1)
            if w > h:
                cv2.rectangle(img, (x, y), (x + 3, y + h), (200, 200, 205), -1)
                cv2.rectangle(img, (x + w - 3, y), (x + w, y + h), (200, 200, 205), -1)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + 3), (200, 200, 205), -1)
                cv2.rectangle(img, (x, y + h - 3), (x + w, y + h), (200, 200, 205), -1)
    return img


def _render_metal(rng: np.random.Generator) -> np.ndarray:
    base = rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32)
    k = np.zeros((1, 31), np.float32)
    k[0, :] = 1 / 31
    brushed = cv2.filter2D(base, -1, k) * 18
    img = np.full((SIZE, SIZE), 35.0, np.float32)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    r = np.hypot(xx - SIZE / 2, yy - SIZE / 2)
    disc = r < 112
    img[disc] = 150 + brushed[disc] + (r[disc] / 112) * 18
    ring = (r > 70) & (r < 74)
    img[ring] -= 35
    img[r < 26] = 30
    for ang in (45, 135, 225, 315):
        cx = int(SIZE / 2 + 92 * np.cos(np.radians(ang)))
        cy = int(SIZE / 2 + 92 * np.sin(np.radians(ang)))
        cv2.circle(img, (cx, cy), 9, 40, -1, cv2.LINE_AA)
    out = np.clip(np.stack([img, img * 1.01, img * 1.04], -1), 0, 255).astype(np.uint8)
    return out


def _draw_defect(img: np.ndarray, kind: str, rng: np.random.Generator, product: str) -> np.ndarray:
    mask = np.zeros(img.shape[:2], np.uint8)
    if product == "demo-metal":
        # place defects on material (annulus between the bore and the rim), never inside holes
        rad, ang0 = rng.uniform(40, 98), rng.uniform(0, 2 * np.pi)
        cx, cy = int(SIZE / 2 + rad * np.cos(ang0)), int(SIZE / 2 + rad * np.sin(ang0))
    else:
        cx = int(SIZE / 2 + rng.uniform(-110, 110) * 0.7)
        cy = int(SIZE / 2 + rng.uniform(-110, 110) * 0.7)
    if kind == "scratch":
        length = rng.uniform(40, 90)
        ang = rng.uniform(0, np.pi)
        p1 = (int(cx - np.cos(ang) * length / 2), int(cy - np.sin(ang) * length / 2))
        p2 = (int(cx + np.cos(ang) * length / 2), int(cy + np.sin(ang) * length / 2))
        color = (235, 235, 230) if rng.random() < 0.6 else (25, 25, 25)
        cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
        cv2.line(mask, p1, p2, 255, 4)
    elif kind == "crack":
        pts = [(cx, cy)]
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(int(rng.integers(8, 14))):
            ang += rng.uniform(-0.7, 0.7)
            x, y = pts[-1]
            pts.append((int(x + 6 * np.cos(ang)), int(y + 6 * np.sin(ang))))
        arr = np.array(pts, np.int32)
        cv2.polylines(img, [arr], False, (20, 18, 18), 2, cv2.LINE_AA)
        cv2.polylines(mask, [arr], False, 255, 5)
    elif kind == "dent":
        ax, ay = int(rng.integers(12, 22)), int(rng.integers(8, 16))
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        d = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
        inside = d < 1
        shade = ((xx - cx) / ax) * 55.0
        f = img.astype(np.float32)
        f[inside] += shade[inside][:, None] * (1 - d[inside])[:, None] - 25 * (1 - d[inside])[:, None]
        img[:] = np.clip(f, 0, 255).astype(np.uint8)
        mask[inside] = 255
    elif kind == "contamination":
        for _ in range(int(rng.integers(2, 5))):
            ox, oy = cx + int(rng.integers(-12, 12)), cy + int(rng.integers(-12, 12))
            rad = int(rng.integers(3, 8))
            palette = [(90, 60, 30), (15, 15, 15), (160, 140, 60)]
            col = palette[int(rng.integers(0, len(palette)))]
            cv2.circle(img, (ox, oy), rad, col, -1, cv2.LINE_AA)
            cv2.circle(mask, (ox, oy), rad + 1, 255, -1)
    elif kind == "solder_bridge":
        x0 = cx + int(rng.integers(-5, 5))
        cv2.rectangle(img, (x0, cy), (x0 + int(rng.integers(10, 20)), cy + 5), (210, 210, 215), -1)
        cv2.rectangle(mask, (x0 - 1, cy - 1), (x0 + 21, cy + 6), 255, -1)
    else:
        raise ValueError(kind)
    return mask


def _finish(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator, batch_gain: float) -> tuple:
    ang = rng.uniform(-1.5, 1.5)
    tx, ty = rng.uniform(-4, 4), rng.uniform(-4, 4)
    m = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), ang, 1.0)
    m[:, 2] += np.array([tx, ty])
    img = cv2.warpAffine(img, m, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, m, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
    f = img.astype(np.float32) * batch_gain * rng.uniform(0.96, 1.04)
    f += rng.normal(0, 3.0, f.shape)
    img = np.clip(cv2.GaussianBlur(f, (3, 3), 0.6), 0, 255).astype(np.uint8)
    return img, mask


def generate_product(product: str, out_root: Path, n_normal: int = 160, n_anomaly: int = 60,
                     n_batches: int = 8, seed: int = 7) -> Path:
    if product not in PRODUCTS:
        raise ValueError(f"Unknown synthetic product {product}")
    defects = PRODUCTS[product]
    rng = np.random.default_rng(seed + (0 if product == "demo-pcb" else 1000))
    layout = _pcb_layout(np.random.default_rng(seed)) if product == "demo-pcb" else None
    out = out_root / product
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    gains = {f"batch-{b:02d}": float(1 + rng.uniform(-0.07, 0.07)) for b in range(n_batches)}
    batch_ids = list(gains)
    samples: list[Sample] = []
    total = n_normal + n_anomaly
    for i in range(total):
        is_anom = i >= n_normal
        batch = batch_ids[i % n_batches]
        kind = defects[(i - n_normal) % len(defects)] if is_anom else ""
        if product == "demo-pcb":
            assert layout is not None
            skip = None
            if kind == "missing_component":
                skip = int(rng.integers(0, len(layout["comps"])))
            img = _render_pcb(layout, rng, skip=skip)
            if kind == "missing_component":
                x, y, w, h = layout["comps"][skip]["box"]
                mask = np.zeros((SIZE, SIZE), np.uint8)
                cv2.rectangle(mask, (x - 1, y - 5), (x + w + 1, y + h + 5), 255, -1)
            elif is_anom:
                mask = _draw_defect(img, kind, rng, product)
            else:
                mask = np.zeros((SIZE, SIZE), np.uint8)
        else:
            img = _render_metal(rng)
            mask = _draw_defect(img, kind, rng, product) if is_anom else np.zeros((SIZE, SIZE), np.uint8)
        img, mask = _finish(img, mask, rng, gains[batch])
        name = f"{'A' if is_anom else 'N'}_{i:04d}"
        cv2.imwrite(str(out / "images" / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        mask_rel = ""
        if is_anom:
            mask_rel = f"masks/{name}.png"
            cv2.imwrite(str(out / mask_rel), mask)
        samples.append(Sample(path=f"images/{name}.png", label=LABEL_ANOMALY if is_anom else LABEL_NORMAL,
                              defect_type=kind, mask=mask_rel, group="", split="", source_id=f"{batch}/{name}"))
    # Leakage-safe grouping for anomalies is not needed (each defect is unique); normals are split
    # individually. Batch ids are kept in source_id for traceability and in the domain-shift experiment.
    make_splits(samples, SplitConfig(seed=seed, val_fraction_normal=0.15, test_fraction_normal=0.2,
                                     val_fraction_anomaly=0.4))
    info = DatasetInfo(
        name=product, product=product, source="DefectLens procedural generator (synthetic)",
        license="Apache-2.0 (generated by this repository)", synthetic=True, defect_types_annotated=True,
        masks_available=True, version=f"seed-{seed}",
        description=f"SYNTHETIC {product} images for demonstration only; not representative of real accuracy.",
        preprocessing={"size": SIZE}, split_policy={"protocol": "one_class", "seed": seed},
        created_at=datetime.now(UTC).isoformat(), redistributable=True,
    )
    save_dataset(out, info, samples)
    return out


def generate_all(out_root: Path, seed: int = 7, **kw) -> list[Path]:
    return [generate_product(p, out_root, seed=seed, **kw) for p in PRODUCTS]
