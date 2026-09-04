"""Compute which ComfyUI versions satisfy comfy-env's contract.

The contract is data, so the supported range does not have to be asserted
by hand and go stale. This evaluates it STATICALLY against any number of
ComfyUI trees: no imports, no torch, no GPU, so it can run anywhere and on
every commit.

Usage:

    sweep_contract.py <comfyui-tree> [<comfyui-tree> ...]
    sweep_contract.py --history <clone> --before 2026-09-01 2026-08-01 ...

Static resolution deliberately answers a narrower question than the runtime
check: it asks whether a module defines a name at module scope, which is
what a rename or a move breaks. It cannot see a symbol that only exists at
runtime, so a MISSING verdict here is evidence and a PRESENT one is the
absence of evidence to the contrary.
"""

import ast
import os
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import harness as H  # noqa: E402

H.bootstrap()

from comfy_env import contract as C  # noqa: E402


def module_path(tree, module):
    parts = module.split(".")
    base = os.path.join(tree, *parts)
    for candidate in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(candidate):
            return candidate
    return None


def defined_names(path):
    """Every name a module binds at module scope."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return None
    names = set()

    def walk(body):
        # Descend through blocks that RUN at import (if/try/with/for) but not
        # into functions or classes, whose names are local. Missing this is
        # why an early version reported comfy.cli_args.args absent from every
        # version: it is assigned inside a module-level conditional.
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For,
                                   ast.While)):
                walk(node.body)
                walk(getattr(node, "orelse", []))
                walk(getattr(node, "finalbody", []))
                for handler in getattr(node, "handlers", []):
                    walk(handler.body)

    walk(tree.body)
    return names


def evaluate_tree(tree, tiers):
    """(ok, failures, notes) for one ComfyUI tree."""
    present = {}
    cache = {}
    for key in C.required_keys(C.BOTH, tiers):
        module, _, attr = key.rpartition(".")
        if module not in cache:
            path = module_path(tree, module)
            cache[module] = defined_names(path) if path else None
        names = cache[module]
        if names is None:
            present[key] = None       # not part of ComfyUI, e.g. comfy_aimdo
        else:
            present[key] = attr in names
    return C.evaluate(present, C.BOTH, tiers)


def report(label, tree):
    print("  {}".format(label))
    for tier_name, tiers in (("floor", (C.FLOOR,)),
                             ("paged", (C.FLOOR, C.PAGED)),
                             ("shared", (C.FLOOR, C.PAGED, C.SHARED))):
        ok, failures, notes = evaluate_tree(tree, tiers)
        missing = len(failures) + len(notes)
        mark = "ok " if ok and not notes else ("DEGRADED" if ok else "FATAL")
        detail = ""
        if missing:
            first = (failures or notes)[0]
            detail = "  <- " + first.split(":")[0]
        print("    {:<7} {:<9}{}".format(tier_name, mark, detail))


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "--history":
        clone, dates = argv[1], argv[3:]
        assert argv[2] == "--before", "expected --before"
        import tempfile
        for date in dates:
            sha = subprocess.run(
                ["git", "-C", clone, "log", "-1",
                 "--before={} 23:59:59".format(date), "--format=%h"],
                capture_output=True, text=True).stdout.strip()
            if not sha:
                print("  {}: no commit".format(date))
                continue
            work = tempfile.mkdtemp(prefix="sweep-")
            subprocess.run(["git", "-C", clone, "worktree", "add", "-q",
                            "--detach", work, sha], check=False)
            try:
                report("{} ({})".format(date, sha), work)
            finally:
                subprocess.run(["git", "-C", clone, "worktree", "remove",
                                "--force", work], check=False)
        return 0
    for tree in argv:
        report(tree, tree)
    return 0


if __name__ == "__main__":
    print("== contract satisfaction by ComfyUI version ==")
    sys.exit(main(sys.argv[1:]))
