"""Contract: the per-call _call_state protocol, which three copies got wrong.

Three of the seven defects found in the 2026-08 transport review were the same
shape -- an invariant maintained in three places (call_method / call_module /
echo), two right and one wrong:

  * echo() never set `worker_pool`, so a PoolIPC reply arrived with no pool
    handle. verify_transport() uses echo as its ORACLE, so its bare except
    swallowed the resulting error and Pool IPC demoted itself permanently on
    every worker -- reported as a routine capability probe.
  * All three `finally` blocks cleared `gpu_demoted` with the constant False
    rather than restoring it, while _lock is an RLock *specifically* so
    VRAM-eviction callbacks can re-enter on the same thread.

These tests pin the protocol itself rather than any one caller, so a fourth
entry point cannot be added with two thirds of it. No torch, no GPU, no
subprocess -- deliberately, because the GPU lane that would have caught the
first bug does not exist (pytest.mark.gpu is declared and applied to nothing).
"""

import threading

from comfy_env.isolation.workers import _ipc_parent
from comfy_env.isolation.workers.subprocess import (
    _enter_call_scope,
    _exit_call_scope,
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


def test_every_rpc_entry_point_uses_the_scope():
    """A fourth caller must not be able to hand-roll two thirds of the protocol.

    This is the check that would have caught the echo bug: it asserts the
    protocol is centralised, not that any one caller happens to be correct.
    """
    import inspect

    from comfy_env.isolation.workers import subprocess as sp

    for name in ("call_method", "call_module", "echo"):
        src = inspect.getsource(getattr(sp.SubprocessWorker, name))
        assert "_enter_call_scope" in src, f"{name} does not enter the call scope"
        assert "_exit_call_scope" in src, f"{name} does not exit the call scope"
        # And nobody reintroduces the constant-assignment form.
        assert "gpu_demoted = False" not in src, (
            f"{name} assigns gpu_demoted a constant instead of restoring it"
        )


def test_worker_model_registry_mutations_hold_the_lock():
    """Bug: `_model_counter[0] += 1` was an unlocked read-modify-write.

    _hooked_to/_hooked_cuda replace torch.nn.Module.to/.cuda GLOBALLY, so they
    fire on whatever thread a pack uses. Two models reaching CUDA concurrently
    could mint the same model_id, leaving the loser GPU-resident with no ledger
    entry -- permanently un-evictable.

    The registrars live inside main()'s closure and cannot be imported, so this
    asserts the structure: every counter mutation sits inside `with
    _registry_lock:`. A structural test is weaker than a racing one, but it is
    the strongest thing available for code that is shipped as source text and
    executed by a foreign interpreter (ADR-0006).
    """
    import ast
    from pathlib import Path

    import comfy_env.isolation.workers as pkg

    src = (Path(pkg.__file__).parent / "_persistent_worker.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

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
