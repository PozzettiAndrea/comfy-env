"""The stale sibling pin warning, and nothing else.

**This module is a deletion unit.** It exists in its own file so that removing
it is `git rm` plus one import line, and so that nothing else grows into it.

What it does: walks every sibling pack under `custom_nodes/` at install time
and warns when one pins a `comfy-env` version older than the installed one.
Warn only. It never fails an install and never changes anything on disk.

Why it exists, and when it goes: comfy-env is installed into the SHARED
ComfyUI host env, which ADR-0022 records as the one exception to comfy-env's
own host-env principle. That means any pack's `pip install -r
requirements.txt` can downgrade comfy-env for every pack on the machine, and
nothing observes it happening. This warning cannot prevent that; it exists so
that when it does happen there is one artifact naming which pack did it.

ADR-0022 plans the repair: split comfy-env so a stale pin can downgrade only
a thin shim rather than everything. **When that ships, delete this file, its
import in `install/__init__.py`, `tests/test_sibling_pins.py`, and the note
in `docs/comfy-env/install.md`.** Do not extend it in the meantime.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Tuple


# comfy-env requirement line in a sibling pack's requirements.txt, e.g.
# "comfy-env==0.3.9" or "comfy_env >=0.3, <0.5". Captures the spec part.
_COMFY_ENV_REQ_RE = re.compile(
    r"^comfy[-_]env\s*(?P<spec>[=<>!~][^#;]*)?", re.IGNORECASE)


def check_sibling_comfy_env_pins(
    node_dir: Path, log: Callable[[str], None] = print,
) -> List[Tuple[str, str]]:
    """Warn when a sibling pack's requirements.txt pins an older comfy-env.

    The shared main env has one comfy-env; whichever pack (re)installs its
    requirements last wins. A stale ``comfy-env==0.3.x`` (or an upper bound
    below the installed version) in ANY pack silently downgrades comfy-env
    for every pack on the next reinstall. Warn-only: never fails an install.

    Returns [(pack_name, offending_line), ...] for testability.
    """
    from ..packages.cuda_wheels import _version_key

    try:
        from .. import __version__ as installed
    except Exception:
        return []
    installed_key = _version_key(installed)

    custom_nodes_dir = Path(node_dir).parent
    findings: List[Tuple[str, str]] = []
    try:
        siblings = sorted(p for p in custom_nodes_dir.iterdir() if p.is_dir())
    except OSError:
        return []

    for sib in siblings:
        if sib.name == Path(node_dir).name:
            continue
        req_file = sib / "requirements.txt"
        if not req_file.is_file():
            continue
        try:
            lines = req_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for raw in lines:
            line = raw.split("#", 1)[0].strip()
            m = _COMFY_ENV_REQ_RE.match(line)
            if not m or not m.group("spec"):
                continue
            stale = False
            for clause in m.group("spec").split(","):
                clause = clause.strip()
                # ~= is deliberately not flagged: `~=0.4.10` still allows
                # newer patch releases, so it is not a downgrade pin.
                cm = re.match(r"(==|<=|<)\s*([0-9][0-9a-zA-Z.\-_+]*)", clause)
                if not cm:
                    continue
                op, ver = cm.group(1), cm.group(2)
                ver_key = _version_key(ver)
                if op in ("==", "<=") and ver_key < installed_key:
                    stale = True
                elif op == "<" and ver_key <= installed_key:
                    stale = True
            if stale:
                findings.append((sib.name, line))

    for pack, line in findings:
        log(f"[comfy-env] WARNING: {pack}/requirements.txt pins '{line}' but "
            f"comfy-env {installed} is installed. If that pack reinstalls its "
            f"requirements, comfy-env will be DOWNGRADED for every pack -- "
            f"update {pack} (or relax its pin).")
    return findings
