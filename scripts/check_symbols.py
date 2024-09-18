"""Static cross-module symbol check: every `from X import Y` and `alias.attr` on imported project modules resolves."""

import ast
import importlib
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(root))
bad = []
for f in sorted((root / "app").rglob("*.py")) + sorted((root / "defectml").rglob("*.py")):
    tree = ast.parse(f.read_text())
    aliases = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.level == 0
            and node.module.split(".")[0] in ("app", "defectml")
        ):
            mod = importlib.import_module(node.module)
            for a in node.names:
                if not hasattr(mod, a.name):
                    try:
                        sub = importlib.import_module(f"{node.module}.{a.name}")
                        aliases[a.asname or a.name] = sub
                    except ImportError:
                        bad.append(f"{f.relative_to(root)}:{node.lineno} {node.module}.{a.name}")
                else:
                    obj = getattr(mod, a.name)
                    if type(obj).__name__ == "module":
                        aliases[a.asname or a.name] = obj
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in ("app", "defectml") and a.asname:
                    aliases[a.asname] = importlib.import_module(a.name)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
            and not hasattr(aliases[node.value.id], node.attr)
        ):
            bad.append(f"{f.relative_to(root)}:{node.lineno} {node.value.id}.{node.attr}")
print("\n".join(sorted(set(bad))) or "all symbols resolve")
sys.exit(1 if bad else 0)
