"""macOS: Dedupe libomp.dylib copies in a site-packages to prevent OpenMP runtime conflicts."""

import glob
import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional, Tuple


class LibompResult(NamedTuple):
    """What one dedupe pass actually did.

    Returned so callers can say what happened instead of announcing success
    unconditionally. A pass that touches nothing is the interesting case:
    every copy skipped by the torch guard is reported with its path, because
    the guard is a substring test (`"torch" in path`) and an env whose own
    directory name carries the torch ABI tag would match it for every
    candidate. If that is what is happening, this log shows it.
    """

    status: str            # "ok" | "not-darwin" | "no-torch-import" | "no-canonical"
    candidates: int = 0    # redundant copies matched by the glob patterns
    linked: int = 0        # copies replaced with a symlink to torch's
    already: int = 0       # copies already pointing at torch's
    skipped: int = 0       # copies the torch guard excluded
    failed: int = 0        # copies that could not be replaced (OSError)
    skipped_paths: Tuple[str, ...] = ()

    @property
    def touched(self) -> bool:
        return bool(self.linked or self.already)

    def summary(self) -> str:
        if self.status == "not-darwin":
            return "not macOS, nothing to do"
        if self.status == "no-torch-import":
            return "torch not importable, nothing to dedupe against"
        if self.status == "no-canonical":
            return "no torch/lib/libomp.dylib to point at -- NOTHING DONE"
        if not self.candidates:
            return "no redundant libomp.dylib found"
        parts = ["{} candidate(s)".format(self.candidates),
                 "{} linked".format(self.linked)]
        if self.already:
            parts.append("{} already linked".format(self.already))
        if self.skipped:
            parts.append("{} skipped as torch's own".format(self.skipped))
        if self.failed:
            parts.append("{} FAILED".format(self.failed))
        line = ", ".join(parts)
        if not self.touched:
            line += " -- NOTHING DEDUPED"
        return line


def dedupe_libomp(site_packages: Optional[Path] = None) -> LibompResult:
    """Symlink redundant libomp.dylib copies to torch's, in `site_packages`.

    macOS only. Many pip wheels ship their own bundled libomp.dylib (torch,
    sklearn, pymeshlab, etc.); having multiple copies loaded in one process
    can corrupt OMP runtime state and SIGSEGV inside native filters.

    With no argument: dedupes the parent process's torch site-packages
    (legacy ComfyUI prestartup behavior).
    With `site_packages` set: dedupes that directory's wheels and the
    enclosing env's `lib/libomp.dylib` (conda-forge libomp at env root).

    The canonical libomp is `<sp>/torch/lib/libomp.dylib` -- usually present
    because torch is installed in every env via the `comfyui` feature.

    Returns a LibompResult describing what was done; see that class for why
    the no-op case is reported in detail rather than silently.
    """
    if sys.platform != "darwin":
        return LibompResult("not-darwin")

    if site_packages is None:
        try:
            import torch
        except ImportError:
            return LibompResult("no-torch-import")
        sp_dir = os.path.dirname(os.path.dirname(torch.__file__))
    else:
        sp_dir = str(site_packages)

    torch_libomp = os.path.join(sp_dir, "torch", "lib", "libomp.dylib")
    if not os.path.exists(torch_libomp):
        return LibompResult("no-canonical")  # No canonical libomp to point at; bail.

    patterns = [
        os.path.join(sp_dir, "*", "Frameworks", "libomp.dylib"),
        os.path.join(sp_dir, "*", ".dylibs", "libomp.dylib"),
        os.path.join(sp_dir, "*", "lib", "libomp.dylib"),
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    # Also handle the env-root libomp at <env>/lib/libomp.dylib. Site-packages
    # lives at <env>/lib/python*/site-packages, so the env-root lib dir is two
    # parents up from sp_dir.
    env_lib_dir = Path(sp_dir).parent.parent  # <env>/lib
    env_libomp = env_lib_dir / "libomp.dylib"
    if env_libomp.exists():
        candidates.append(str(env_libomp))

    linked = already = skipped = failed = 0
    skipped_paths = []

    for libomp in candidates:
        if "torch" in libomp:
            skipped += 1
            skipped_paths.append(libomp)
            continue
        try:
            if os.path.islink(libomp):
                if os.path.realpath(libomp) == os.path.realpath(torch_libomp):
                    already += 1
                    continue
                os.unlink(libomp)
            else:
                os.rename(libomp, libomp + ".bak")
            os.symlink(torch_libomp, libomp)
            linked += 1
        except OSError:
            failed += 1

    return LibompResult(
        "ok",
        candidates=len(candidates),
        linked=linked,
        already=already,
        skipped=skipped,
        failed=failed,
        skipped_paths=tuple(skipped_paths),
    )
