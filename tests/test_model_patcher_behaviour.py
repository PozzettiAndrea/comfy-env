"""Contract: the subprocess model proxy never lies and never raises upward.

Two rules the previous ModelPatcher subclass broke, both of which corrupt
ComfyUI's memory manager rather than merely misbehaving locally:

* `partially_unload` must return the bytes the worker ACTUALLY freed.
  `LoadedModel.model_unload` compares `freed >= memory_to_free`; a too-large
  return tells ComfyUI "satisfied, keep me resident" while the weights are
  gone, and disables its escalation to full eviction.
* Eviction paths must not raise. They execute inside `free_memory`'s loop; an
  exception there escapes ComfyUI's memory manager and poisons every
  subsequent load for the life of the process.

Runs without ComfyUI/torch by stubbing `comfy.model_management`.
"""

import sys
import types

import pytest


@pytest.fixture()
def patcher_mod(monkeypatch):
    """Import isolation.model_patcher with comfy.model_management stubbed."""
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: types.SimpleNamespace(type="cpu")
    mm.get_total_memory = lambda dev: 0
    mm.get_free_memory = lambda dev: 0
    comfy_pkg = types.ModuleType("comfy")
    comfy_pkg.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    for name in list(sys.modules):
        if name.endswith("isolation.model_patcher"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    from comfy_env.isolation import model_patcher
    return model_patcher


class FakeWorker:
    """Records commands; can play dead or fail."""

    def __init__(self, alive=True, reply=None, boom=False):
        import threading
        self._alive, self._reply, self._boom = alive, reply or {}, boom
        self.sent = []
        self._mem_lock = threading.Lock()   # the real worker's leaf mutex
        self._calls_in_flight = 0

    def is_alive(self):
        return self._alive

    def send_command(self, command, **kwargs):
        self.sent.append((command, kwargs))
        if self._boom:
            raise RuntimeError("transport exploded")
        return dict(self._reply)


def _make(patcher_mod, worker, size=8 * 1024**3, resident=None):
    p = patcher_mod.SubprocessModelPatcher(
        worker=worker, worker_generation=1, model_id="m1", model_size=size,
        load_device="cuda:0", offload_device="cpu")
    p.model.model_loaded_weight_memory = size if resident is None else resident
    p.model.device = "cuda:0"
    return p


def test_partial_unload_returns_actual_freed_not_size(patcher_mod):
    """The worker freed 200MB of an 8GB model -> report 200MB, not 8GB."""
    freed = 200 * 1024**2
    w = FakeWorker(reply={"freed": freed, "resident": 8 * 1024**3 - freed})
    p = _make(patcher_mod, w)
    got = p.partially_unload("cpu", memory_to_free=freed)
    assert got == freed, "must report what was actually freed"
    assert got != p.size, "returning self.size defeats ComfyUI's escalation"
    assert p.loaded_size() == 8 * 1024**3 - freed
    assert p.current_loaded_device() == "cuda:0", "still partly resident"


def test_short_return_is_allowed(patcher_mod):
    """Freeing less than asked is the designed path -- ComfyUI escalates."""
    w = FakeWorker(reply={"freed": 50, "resident": 8 * 1024**3 - 50})
    p = _make(patcher_mod, w)
    assert p.partially_unload("cpu", memory_to_free=1024**3) == 50


def test_full_unload_marks_offloaded(patcher_mod):
    w = FakeWorker(reply={"freed": 8 * 1024**3, "resident": 0})
    p = _make(patcher_mod, w)
    p.partially_unload("cpu", memory_to_free=8 * 1024**3)
    assert p.loaded_size() == 0
    assert p.current_loaded_device() == "cpu"


def test_unload_of_already_offloaded_is_free(patcher_mod):
    """No IPC round trip when there is nothing resident."""
    w = FakeWorker()
    p = _make(patcher_mod, w, resident=0)
    assert p.partially_unload("cpu", memory_to_free=1024) == 0
    assert w.sent == [], "must not send a command for a no-op"


def test_dead_worker_does_not_raise_from_eviction(patcher_mod):
    """A dead worker means the VRAM is already gone -- report it freed."""
    resident = 4 * 1024**3
    p = _make(patcher_mod, FakeWorker(alive=False), resident=resident)
    assert p.partially_unload("cpu", memory_to_free=1024) == resident
    assert p.loaded_size() == 0
    p.detach()  # must also not raise
    assert p.current_loaded_device() == "cpu"


def test_live_worker_that_refuses_an_unload_is_not_reported_as_offloaded(patcher_mod):
    """A failed command on a LIVE worker means the weights are still resident.

    FakeWorker(boom=True) is alive; only the command fails. Collapsing that
    into the dead-worker outcome told ComfyUI it had reclaimed N GB it had
    not, and zeroed loaded_size so the model was never picked for eviction
    again -- every later admission decision computed against a card believed
    to have N GB more free. Rule #1 of this module: never lie about bytes.
    """
    resident = 8 * 1024**3
    p = _make(patcher_mod, FakeWorker(boom=True), resident=resident)

    freed = p.partially_unload("cpu", memory_to_free=1024)   # must not raise
    assert freed == 0, "reported bytes freed that are still on the card"
    assert p.loaded_size() == resident, (
        "loaded_size was zeroed, so ComfyUI will never try to evict this again"
    )

    p.detach()                                               # must not raise
    assert p.loaded_size() == resident
    assert p.current_loaded_device() != "cpu", (
        "reported the model as moved off the GPU after the move failed"
    )


def test_load_path_does_raise_on_dead_worker(patcher_mod):
    """Loading is not eviction: failing loudly is correct there."""
    p = _make(patcher_mod, FakeWorker(alive=False), resident=0)
    with pytest.raises(RuntimeError, match="no longer available"):
        p.partially_load("cuda:0", extra_memory=1024)


def test_detach_is_idempotent(patcher_mod):
    w = FakeWorker(reply={})
    p = _make(patcher_mod, w, resident=0)
    p.model.device = "cpu"
    p.detach()
    p.detach()
    assert w.sent == [], "already offloaded -> no redundant IPC"


def test_partial_load_reports_actual_loaded(patcher_mod):
    loaded = 3 * 1024**3
    w = FakeWorker(reply={"loaded": loaded, "resident": loaded})
    p = _make(patcher_mod, w, resident=0)
    p.model.device = "cpu"
    assert p.partially_load("cuda:0", extra_memory=loaded) == loaded
    assert p.loaded_size() == loaded
    assert p.current_loaded_device() == "cuda:0"


def test_patching_raises_instead_of_silently_dropping(patcher_mod):
    """Storing patches that never apply would mean wrong output, no error."""
    p = _make(patcher_mod, FakeWorker())
    with pytest.raises(NotImplementedError, match="subprocess"):
        p.add_patches({}, 1.0)
    with pytest.raises(NotImplementedError, match="cloned"):
        p.clone()


def test_unknown_attribute_names_itself(patcher_mod):
    """Upstream drift must produce a pointed traceback, not silence."""
    p = _make(patcher_mod, FakeWorker())
    with pytest.raises(AttributeError, match="some_new_upstream_field"):
        _ = p.some_new_upstream_field
    # dunder probes stay quiet so copy/pickle/weakref protocols work
    with pytest.raises(AttributeError):
        _ = p.__deepcopy__


def test_inner_model_is_weakref_able(patcher_mod):
    """LoadedModel.model_load takes weakref.finalize on .model."""
    import weakref
    p = _make(patcher_mod, FakeWorker())
    assert weakref.ref(p.model)() is p.model
    assert weakref.ref(p)() is p       # LoadedModel._set_model weakrefs the patcher
