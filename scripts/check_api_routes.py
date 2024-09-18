"""Cross-check that every /api URL used by the frontend matches a FastAPI route (method-agnostic).

Run from the repo root:  backend/.venv/bin/python scripts/check_api_routes.py
"""

import os
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "backend"))
os.environ.setdefault("DL_WARMUP_MODELS", "false")
from app.main import app  # noqa: E402

templates = [re.sub(r"\{[^}]+\}", "X", p) for p in app.openapi()["paths"] if p.startswith("/api")]
templates.append("/api/auth")  # api.ts prefix check (startsWith), not a request


def matches(url: str) -> bool:
    """X in either the URL (a ${...} expression) or the template ({param}) matches any one segment."""
    u, n = url.split("/"), None
    for t in templates:
        n = t.split("/")
        if len(n) == len(u) and all(a == b or a == "X" or b == "X" for a, b in zip(u, n, strict=True)):
            return True
    return False


bad = 0
pat = re.compile(r"""[`"'](/api/[^`"'\s]*)""")
for f in sorted((root / "frontend" / "src").rglob("*.ts*")):
    for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        for m in pat.finditer(line):
            url = m.group(1)
            url = url.split("${qs", 1)[0].split("${product", 1)[0]  # query-string helpers
            url = re.sub(r"\$\{[^}]*\}?", "X", url).split("?")[0]  # ${id} -> X
            url = url.rstrip("/") or url
            if not matches(url):
                bad += 1
                print(f"{f.relative_to(root)}:{n}: no route for {url}")
print("all frontend API URLs resolve" if not bad else f"{bad} unresolved")
sys.exit(1 if bad else 0)
