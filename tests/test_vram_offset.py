"""Contract: admission survives a blind `get_free_memory`.

On Windows/WDDM `torch.cuda.mem_get_info` reports the CALLING PROCESS's
budget, not device-wide free -- measured: a sibling allocated 13.0 GiB while
the parent's view moved 75 MB. ComfyUI decides how much to evict from
`memory_required - get_free_memory(device)`, so with that number stuck near
full-card the difference goes negative and `free_memory` evicts NOTHING.

The fix pre-compensates: add the parent's over-report (= worker-held bytes) to
the target passed to `free_memory`. The offset is constant across the eviction
loop, and parent-side unloads move the blind and true numbers together, so
ComfyUI's own arithmetic behaves as if it could see the whole device.

These tests pin that arithmetic and the zero-dependency fallback.
"""

import sys
import types

import pytest

GB = 1024 ** 3


@pytest.fixture()
def pool_mod(monkeypatch):
    """Import isolation.pool with a stubbed comfy.model_management."""
    calls = {"free_memory": []}

    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: types.SimpleNamespace(type="cuda", index=0)
    mm._blind_free = 15 * GB          # what a blind mem_get_info reports
    mm.get_free_memory = lambda dev: mm._blind_free
    mm.minimum_inference_memory = lambda: 1 * GB
    mm.free_memory = lambda amount, dev, **kw: calls["free_memory"].append(amount)
    mm.vram_state = types.SimpleNamespace(name="NORMAL_VRAM", value=3)
    mm.VRAMState = types.SimpleNamespace(LOW_VRAM=types.SimpleNamespace(value=1))
    mm.EXTRA_RESERVED_VRAM = 0
    comfy_pkg = types.ModuleType("comfy")
    comfy_pkg.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    from comfy_env.isolation import pool
    pool._WORKER_PATCHERS.clear()
    return pool, mm, calls


class FakePatcher:
    def __init__(self, resident):
        self._resident = resident

    def loaded_size(self):
        return self._resident


def test_offset_compensates_the_blind_view(pool_mod, monkeypatch):
    """Parent believes 15GB free; a worker really holds 13GB. The target passed
    to free_memory must be inflated by that 13GB, or nothing is ever evicted."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: 2 * GB)

    pool._handle_vram_budget({"total_size": 4 * GB})

    assert calls["free_memory"], "free_memory must be called"
    asked = calls["free_memory"][0]
    offset = 15 * GB - 2 * GB
    need = int(4 * GB * pool._REQUEST_SLACK) + pool._WORKER_FIXED_VRAM_COST + 1 * GB
    assert asked == need + offset
    # Without compensation ComfyUI computes need - 15GB < 0 and evicts nothing.
    assert asked > mm._blind_free, (
        "target must exceed the blind free value, or free_memory's "
        "`memory_to_free > 0` guard never passes")


def test_ledger_fallback_when_nvml_unavailable(pool_mod, monkeypatch):
    """No pynvml and no nvidia-smi: reconstruct the offset from comfy-env's own
    books, which already know every worker model's residency."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: None)
    pool._WORKER_PATCHERS["envA"] = {"m1": FakePatcher(6 * GB),
                                     "m2": FakePatcher(2 * GB)}

    pool._handle_vram_budget({"total_size": 1 * GB})

    held = 8 * GB + pool._WORKER_FIXED_VRAM_COST      # one live worker
    need = int(1 * GB * pool._REQUEST_SLACK) + pool._WORKER_FIXED_VRAM_COST + 1 * GB
    assert calls["free_memory"][0] == need + held


def test_headroom_is_additive_not_only_multiplicative(pool_mod, monkeypatch):
    """A per-process CUDA context is a constant, not a percentage: 1.1x on a
    small model does not cover ~250MB of context+handles."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: 15 * GB)

    pool._handle_vram_budget({"total_size": 100 * 1024 ** 2})  # 100MB model

    asked = calls["free_memory"][0]
    assert asked >= 100 * 1024 ** 2 + pool._WORKER_FIXED_VRAM_COST
    assert asked > 100 * 1024 ** 2 * 1.1, "multiplicative-only headroom is too small"


def test_inference_reserve_is_included(pool_mod, monkeypatch):
    """An in-process load reserves minimum_inference_memory; a worker load must
    reserve it too, or worker models get ~1GB less headroom than host ones."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: 15 * GB)
    pool._handle_vram_budget({"total_size": 2 * GB})
    assert calls["free_memory"][0] >= 2 * GB + 1 * GB


def test_worker_receives_true_device_free(pool_mod, monkeypatch):
    """The worker corrects its OWN blindness from this number: its
    get_free_memory minus device_free = what everyone else holds."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: 3 * GB)
    reply = pool._handle_vram_budget({"total_size": 1 * GB})
    assert reply["device_free_bytes"] == 3 * GB


def test_worker_held_bytes_counts_fixed_cost_per_worker(pool_mod):
    pool, mm, calls = pool_mod
    pool._WORKER_PATCHERS["a"] = {"m": FakePatcher(1 * GB)}
    pool._WORKER_PATCHERS["b"] = {"m": FakePatcher(2 * GB)}
    assert pool._worker_held_bytes() == 3 * GB + 2 * pool._WORKER_FIXED_VRAM_COST
