"""Contract: _ipc_shared.py is the one module both sides of the boundary run.

Two rules govern it, and until now nothing enforced either. Every de-duplication
between the parent and the worker pushes more code into this file, so the rules
need a tripwire before the code arrives, not after.

1. It must parse under Python 3.10 -- the floor comfy-env supports, matching
   ComfyUI's own requires-python >= 3.10 (a config pinning lower is rejected at
   load). The worker program is read as TEXT and exec'd by the pack env's
   interpreter (ADR-0006), and _ipc_shared.py is copied next to it, so a
   3.11+ construct here is a SyntaxError in the worker and never in CI.

2. It must not import torch or numpy at module scope. _persistent_worker.py
   imports it at line ~20, BEFORE it pins CPU affinity and before the
   deliberate torch-before-numpy ordering that keeps fbgemm.dll loadable on
   Windows. A module-scope torch/numpy import here silently reorders that.
"""

import ast
from pathlib import Path

import pytest

import comfy_env.isolation.workers as _workers_pkg

WORKERS = Path(_workers_pkg.__file__).parent

# Both files cross the interpreter boundary: the worker as source text, and
# _ipc_shared.py copied alongside it (see subprocess.py's worker staging).
BOUNDARY_FILES = ["_ipc_shared.py", "_persistent_worker.py"]


@pytest.mark.parametrize("name", BOUNDARY_FILES)
def test_parses_under_python_310(name):
    src = (WORKERS / name).read_text(encoding="utf-8")
    try:
        ast.parse(src, feature_version=(3, 10))
    except SyntaxError as e:
        pytest.fail(
            f"{name} uses syntax newer than Python 3.10 at line {e.lineno}: {e.msg}. "
            f"This file is executed by the pack env's interpreter, which may be as "
            f"old as 3.10 -- the failure would appear only at worker startup, "
            f"never in CI."
        )


def test_ipc_shared_has_no_module_scope_torch_or_numpy():
    """Heavy imports stay inside function bodies.

    The worker imports this module before its torch-before-numpy DLL ordering
    runs. Hoisting `import torch` (or numpy) to the top here is invisible on
    Linux and breaks Windows with WinError 127 on fbgemm.dll.
    """
    tree = ast.parse((WORKERS / "_ipc_shared.py").read_text(encoding="utf-8"))
    banned = {"torch", "numpy"}
    offenders = []

    for node in tree.body:                       # module scope only
        if isinstance(node, ast.Import):
            offenders += [(node.lineno, a.name) for a in node.names
                          if a.name.split(".")[0] in banned]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in banned:
                offenders.append((node.lineno, node.module))

    assert not offenders, (
        f"module-scope heavy import(s) in _ipc_shared.py: {offenders}. "
        f"Move them inside the functions that need them."
    )


def test_ipc_shared_imports_nothing_from_comfy_env():
    """ADR-0006: the worker cannot import comfy_env, and it imports this file."""
    tree = ast.parse((WORKERS / "_ipc_shared.py").read_text(encoding="utf-8"))
    bad = [
        (n.lineno, n.module or "." * n.level)
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom)
        and ((n.module or "").startswith("comfy_env") or n.level > 0)
    ] + [
        (n.lineno, a.name) for n in ast.walk(tree)
        if isinstance(n, ast.Import)
        for a in n.names if a.name.startswith("comfy_env")
    ]
    assert not bad, (
        f"_ipc_shared.py reaches into comfy_env at {bad}. It is copied next to "
        f"the worker and imported by its bare basename; comfy_env is not "
        f"importable there."
    )


def test_worker_does_not_shadow_the_shared_ipc_caches():
    """The worker must not define its own IPC forwarding caches.

    It did, at module scope, while _ipc_shared's comment claimed both sides
    shared one copy. The consequence was not a stale read: _evict_cache_if_needed
    and _cleanup_ipc_cache are both parent-only and the worker cannot import
    them, so the worker's copies were never bounded and never swept. Each entry
    holds a strong reference to an imported CUDA mapping, which pins the
    EXPORTER's allocation in the other process -- so a long-lived worker
    permanently drained the parent's VRAM, invisibly to every ledger the system
    has.
    """
    src = (WORKERS / "_persistent_worker.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    shadowed = [
        (node.lineno, t.id)
        for node in tree.body                       # module scope only
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
        and t.id in {"_cuda_ipc_metadata_cache", "_cuda_ipc_cache_tensors"}
    ]
    assert not shadowed, (
        f"worker shadows the shared IPC cache(s) at {shadowed}; its copies are "
        f"never evicted, so they pin the parent's VRAM for the worker's lifetime"
    )
    assert "_ipc_shared._cuda_ipc_metadata_cache" in src, (
        "the worker's serializer must read the shared, bounded cache"
    )
