"""Minimal Prometheus-compatible metrics registry (exposed at /api/metrics)."""

from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters: dict[tuple, float] = defaultdict(float)
        self.sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])  # sum, count

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self.counters[(name, tuple(sorted(labels.items())))] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            s = self.sums[name]
            s[0] += value
            s[1] += 1

    def render(self, gauges: dict[str, float] | None = None) -> str:
        lines = []
        with self._lock:
            for (name, labels), v in sorted(self.counters.items()):
                lab = ",".join(f'{k}="{val}"' for k, val in labels)
                lines.append(f"defectlens_{name}{{{lab}}} {v}" if lab else f"defectlens_{name} {v}")
            for name, (s, c) in sorted(self.sums.items()):
                lines.append(f"defectlens_{name}_sum {s}")
                lines.append(f"defectlens_{name}_count {c}")
        for k, v in (gauges or {}).items():
            lines.append(f"defectlens_{k} {v}")
        return "\n".join(lines) + "\n"


METRICS = Metrics()
