"""Accelerator import lint (static, advisory).

The rule: accelerator packages (an env's `[cuda] packages`) must be imported
LAZILY, inside the nodes that declare that accelerator (ACCELERATOR class
attribute). A top-level import kills the metadata scan on machines where the
package isn't installed, which silently unregisters EVERY node in the env.

Static analysis lies in both directions (importlib tricks, module
__getattr__), so this lint is a cheap dev-time companion to the
authoritative runtime check (the sys.modules delta in the metadata scan) --
not a replacement.

Finding levels:
  error    -- unguarded top-level import of an accelerator package
              (provably fatal to the CPU metadata scan)
  advisory -- top-level import guarded by try/except (survives the scan but
              is still against the lazy-import rule); torch.cuda usage in a
              module whose classes declare no ACCELERATOR (legal
              opportunistic-GPU pattern, worth a look)

Known limitation: matches by package name (dash/underscore normalized). A
distribution whose import name differs (e.g. faithc-aot -> faithcontour)
evades the static match; the runtime delta check covers those via
importlib.metadata.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List

from .config import CONFIG_FILE_NAME, load_config


def _import_names(cuda_packages: List[str]) -> set:
    names = set()
    for pkg in cuda_packages:
        base = str(pkg).strip().lower()
        names.add(base.replace("-", "_"))
    return names


def _module_declares_accelerator(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                targets = []
                if isinstance(stmt, ast.Assign):
                    targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    targets = [stmt.target.id]
                if "ACCELERATOR" in targets:
                    return True
    return False


def _top_level_imports(tree: ast.Module):
    """Yield (stmt, guarded) for module-body imports; guarded = inside try."""
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            yield stmt, False
        elif isinstance(stmt, ast.Try):
            for inner in stmt.body:
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    yield inner, True


def _imported_roots(stmt) -> List[str]:
    if isinstance(stmt, ast.Import):
        return [alias.name.split(".")[0].lower() for alias in stmt.names]
    if isinstance(stmt, ast.ImportFrom) and stmt.module and stmt.level == 0:
        return [stmt.module.split(".")[0].lower()]
    return []


def _uses_torch_cuda(tree: ast.Module) -> int:
    """Line of first torch.cuda attribute access, or 0."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr == "cuda"
                and isinstance(node.value, ast.Name) and node.value.id == "torch"):
            return node.lineno
    return 0


def lint_accelerator_imports(root: Path) -> List[Dict]:
    """Lint every comfy-env.toml-scoped env under root. Returns findings."""
    findings: List[Dict] = []
    root = Path(root)
    for config_path in sorted(root.rglob(CONFIG_FILE_NAME)):
        try:
            cfg = load_config(config_path)
        except Exception:
            continue
        accel_names = _import_names(cfg.cuda_packages)
        env_dir = config_path.parent
        for py in sorted(env_dir.rglob("*.py")):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            rel = py.relative_to(root)
            if accel_names:
                for stmt, guarded in _top_level_imports(tree):
                    hit = [r for r in _imported_roots(stmt) if r in accel_names]
                    if not hit:
                        continue
                    if guarded:
                        findings.append({
                            "level": "advisory", "file": str(rel), "line": stmt.lineno,
                            "message": (f"top-level import of accelerator package "
                                        f"{hit[0]} (guarded by try/except -- survives "
                                        f"the scan, but the rule is lazy imports "
                                        f"inside the declaring node)")})
                    else:
                        findings.append({
                            "level": "error", "file": str(rel), "line": stmt.lineno,
                            "message": (f"top-level import of accelerator package "
                                        f"{hit[0]} -- fatal to the metadata scan on "
                                        f"machines without it; import lazily inside "
                                        f"the ACCELERATOR-declaring node")})
            cuda_line = _uses_torch_cuda(tree)
            if cuda_line and not _module_declares_accelerator(tree):
                findings.append({
                    "level": "advisory", "file": str(rel), "line": cuda_line,
                    "message": ("torch.cuda used but no class in this module "
                                "declares ACCELERATOR -- fine if there is a real "
                                "CPU fallback (opportunistic GPU), otherwise "
                                "declare ACCELERATOR")})
    return findings
