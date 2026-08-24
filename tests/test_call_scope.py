"""Contract: the per-call _ipc_parent._call_state protocol.

Pins the protocol rather than any one caller: call_method, call_module and
echo each maintained it by hand, and echo was missing a third of it.
"""

import ast
import threading
from pathlib import Path

from comfy_env.isolation.workers import _ipc_parent
from comfy_env.isolation.workers.subprocess import (
    _enter_call_scope,
    _exit_call_scope,
)


def _worker_source():
    """The worker program's text. It is exec'd by a foreign interpreter
    (ADR-0006) and cannot be imported, so its scopes are only reachable here."""
    import comfy_env.isolation.workers as pkg

    return (Path(pkg.__file__).parent / "_persistent_worker.py").read_text(
        encoding="utf-8"
    )


class _FakeWorker:
    """Just the two attributes the scope reads."""

    def __init__(self, zero_copy_ok=True, pool="POOL"):
        self.gpu_zero_copy_ok = zero_copy_ok
        self._worker_pool = pool


def _state():
    return (
        getattr(_ipc_parent._call_state, "gpu_demoted", None),
        getattr(_ipc_parent._call_state, "worker_pool", None),
    )


def test_scope_sets_both_halves():
    """The bug: echo set gpu_demoted but not worker_pool."""
    prev = _enter_call_scope(_FakeWorker(zero_copy_ok=False, pool="P1"))
    try:
        demoted, pool = _state()
        assert demoted is True, "gpu_demoted not set from gpu_zero_copy_ok"
        assert pool == "P1", "worker_pool not set -- this is the echo() bug"
    finally:
        _exit_call_scope(prev)


def test_nested_call_restores_outer_state_not_a_constant():
    """The bug: `finally` assigned False instead of restoring.

    _lock is an RLock so a VRAM-eviction callback can re-enter on the SAME
    thread. The inner call's exit must not re-enable zero-copy for an outer
    call the canary had demoted.
    """
    outer = _FakeWorker(zero_copy_ok=False, pool="OUTER")   # demoted
    inner = _FakeWorker(zero_copy_ok=True, pool="INNER")    # not demoted

    outer_prev = _enter_call_scope(outer)
    assert _state() == (True, "OUTER")

    inner_prev = _enter_call_scope(inner)
    assert _state() == (False, "INNER")
    _exit_call_scope(inner_prev)

    # Had the exit assigned constants, this would be (False, None) and the
    # outer call would resume exporting handles the worker cannot import.
    assert _state() == (True, "OUTER"), "inner call clobbered the outer scope"
    _exit_call_scope(outer_prev)


def test_scope_is_thread_local():
    """_call_state is threading.local; one worker's call must not leak to another."""
    seen = {}
    barrier = threading.Barrier(2)

    def run(name, pool, out):
        prev = _enter_call_scope(_FakeWorker(pool=pool))
        try:
            barrier.wait(timeout=5)      # both inside their scope at once
            out[name] = _state()[1]
        finally:
            _exit_call_scope(prev)

    a = threading.Thread(target=run, args=("a", "POOL_A", seen))
    b = threading.Thread(target=run, args=("b", "POOL_B", seen))
    a.start(), b.start()
    a.join(timeout=10), b.join(timeout=10)

    assert seen == {"a": "POOL_A", "b": "POOL_B"}, seen


def test_every_rpc_entry_point_restores_the_scope(monkeypatch):
    """Every entry point must leave _call_state as it found it, win or lose.

    Asserted by observation, not by grepping the source for a call: a shared
    decorator or pushing the scope into _send_request would satisfy this and
    should not have to fail it.
    """
    from comfy_env.isolation.workers import subprocess as sp

    calls = {
        "call_method": (("mod", "Cls", "meth"), {}),
        "call_module": (("mod", "func"), {}),
        "echo": ((), {}),
    }

    for name, (args, kwargs) in calls.items():
        for failing in (False, True):
            worker = sp.SubprocessWorker.__new__(sp.SubprocessWorker)
            worker.name = "scope-worker"
            worker._lock = threading.RLock()
            worker.gpu_zero_copy_ok = False        # demoted, must survive
            worker._worker_pool = "POOL"
            worker._call_counter = 0
            worker._shutdown = False
            worker._last_new_models = []
            worker.default_timeout = 5

            def _send_request(request, timeout):
                assert _state() == (True, "POOL"), (
                    f"{name} did not establish the scope before sending"
                )
                if failing:
                    raise RuntimeError("worker exploded")
                return {"status": "ok", "result": None, "call_id": request.get("call_id")}

            monkeypatch.setattr(worker, "_send_request", _send_request, raising=False)
            monkeypatch.setattr(worker, "_ensure_started", lambda: None, raising=False)

            sentinel = _enter_call_scope(_FakeWorker(zero_copy_ok=True, pool="OUTER"))
            try:
                try:
                    getattr(worker, name)(*args, **kwargs)
                except Exception:
                    if not failing:
                        raise
                assert _state() == (False, "OUTER"), (
                    f"{name} did not restore the outer scope "
                    f"({'exception' if failing else 'success'} path)"
                )
            finally:
                _exit_call_scope(sentinel)


def test_auto_registration_hooks_are_installed_at_worker_startup():
    """0.4.29 regression: a `with _registry_lock:` edit re-indented this block.

    The Module.to/.cuda hooks must be installed in main(), unconditionally, at
    worker startup. d7b8439 moved them inside _register_if_cuda -- which has a
    single caller (the shimmed load_models_gpu) and three early returns above
    the block -- so a pack that does model.cuda() itself registered nothing and
    its VRAM was invisible to the parent's ledger for the process lifetime.

    The worker is shipped as source text and exec'd by a foreign interpreter
    (ADR-0006), so its scopes cannot be introspected any other way. The prior
    version of this test asserted the counter sat inside the lock; the
    regression satisfied that and shipped.
    """
    tree = ast.parse(_worker_source())

    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node

    def enclosing_function(node):
        node = getattr(node, "_parent", None)
        while node is not None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
            node = getattr(node, "_parent", None)
        return "<module>"

    installs = [
        (n.lineno, t.attr, enclosing_function(n))
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Attribute)
        and t.attr in ("to", "cuda")
        and "Module" in ast.dump(t.value)
    ]

    assert len(installs) == 2, f"expected Module.to and Module.cuda hooks, got {installs}"
    for lineno, attr, scope in installs:
        assert scope == "main", (
            f"Module.{attr} hook installed inside {scope}() at line {lineno}, not main(). "
            "Auto-registration is then conditional on that function being reached."
        )


def test_cuda_registration_has_exactly_one_body():
    """The duplication that made the re-indent possible.

    _auto_register_if_cuda and _register_if_cuda held the same eight lines
    twice; the lock was added to both, and the second edit swallowed the code
    below it. One body cannot drift from itself.
    """
    tree = ast.parse(_worker_source())
    fns = {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "_register_cuda_module" in fns, "the shared registration body is gone"

    for name in ("_auto_register_if_cuda", "_register_if_cuda"):
        body = [s for s in fns[name].body if not isinstance(s, ast.Expr)]
        mutates = [
            s for s in ast.walk(fns[name])
            if isinstance(s, ast.AugAssign)
            and isinstance(s.target, ast.Subscript)
            and getattr(s.target.value, "id", None) == "_model_counter"
        ]
        assert not mutates, (
            f"{name} mints ids itself again -- it must delegate to "
            "_register_cuda_module so the invariant has one home"
        )
        assert len(body) <= 2, f"{name} has grown its own registration body again"


def test_model_counter_mutations_hold_the_lock():
    """Bug: `_model_counter[0] += 1` was an unlocked read-modify-write.

    Reached from _hooked_to/_hooked_cuda, which replace torch.nn.Module.to and
    .cuda GLOBALLY and so fire on whatever thread a pack uses. Two models
    reaching CUDA concurrently could mint the same model_id, leaving the loser
    GPU-resident with no ledger entry -- permanently un-evictable.
    """
    tree = ast.parse(_worker_source())

    def mutations_outside_lock(node, inside_lock=False):
        found = []
        for child in ast.iter_child_nodes(node):
            now = inside_lock
            if isinstance(child, ast.With):
                now = inside_lock or any(
                    isinstance(i.context_expr, ast.Name)
                    and i.context_expr.id == "_registry_lock"
                    for i in child.items
                )
            if isinstance(child, ast.AugAssign):
                tgt = child.target
                if (isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "_model_counter"
                        and not now):
                    found.append(child.lineno)
            found += mutations_outside_lock(child, now)
        return found

    unlocked = mutations_outside_lock(tree)
    assert not unlocked, (
        f"_model_counter mutated without _registry_lock at line(s) {unlocked}"
    )
