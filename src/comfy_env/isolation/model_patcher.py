"""Duck-typed proxy presenting subprocess models to ComfyUI's memory manager.

A worker-resident GPU model is registered in ComfyUI's ``current_loaded_models``
as a ``SubprocessModelPatcher`` so it participates in normal VRAM eviction:
ComfyUI decides *when* to evict, comfy-env decides *how* (an IPC command that
moves the real weights inside the worker).

Deliberately **not** a ``comfy.model_patcher.ModelPatcher`` subclass.
Subclassing inherited ~120 members -- ``add_patches``, ``load``,
``apply_hooks``, ``pinned_memory_size`` and friends -- every one of which is
wrong for an object that holds no weights, and none of which were disabled.
The failure mode of that design is *silent*: upstream adds a field or changes a
contract, an inherited method runs against a fake model, and the result is a
wrong number rather than an exception.

This class implements only what ComfyUI actually touches (verified against
``model_management.py`` call sites; see ``tests/test_model_patcher_surface.py``,
which fails when that set changes). Everything else hits ``__getattr__`` and
raises **naming the attribute**, turning future upstream drift into a pointed
traceback instead of silent corruption.

Two honesty rules the old proxy broke:

* **Never lie about bytes.** ``partially_unload`` returns what the worker
  actually freed. Returning a too-large number defeats ComfyUI's escalation
  ladder: ``LoadedModel.model_unload`` compares ``freed >= memory_to_free`` and,
  on a short return, falls through to ``detach()`` and full eviction by itself.
* **Never raise inside someone else's loop.** Eviction paths
  (``partially_unload`` / ``detach``) treat a dead or restarted worker as
  "already offloaded". A raise there propagates out of ``free_memory`` and
  poisons every subsequent load for the life of the process.
"""

import logging
import sys

import comfy.model_management

from ..debug import VRAM as _DBG_VRAM

log = logging.getLogger("comfy_env.model_patcher")


def _log_vram(label: str) -> None:
    """Log compact VRAM state around model load/unload."""
    if not _DBG_VRAM:
        return
    try:
        dev = comfy.model_management.get_torch_device()
        if dev.type != "cuda":
            return
        total = comfy.model_management.get_total_memory(dev) // (1024 * 1024)
        free = comfy.model_management.get_free_memory(dev) // (1024 * 1024)
        used = total - free
        print(f"[comfy-env] [VRAM] {label}: {used} / {total} MB", file=sys.stderr, flush=True)
    except Exception:
        pass


class SubprocessModel:
    """Stand-in for a patcher's ``.model`` (the inner nn.Module).

    Holds no weights -- the real module lives in the worker. MUST stay a plain
    class: ``LoadedModel.model_load`` takes a ``weakref.finalize`` on this
    object, so it has to be weak-referenceable.
    """

    def __init__(self, size_bytes, device):
        self.device = device
        self.model_loaded_weight_memory = 0
        self.model_lowvram = False
        self.lowvram_patch_counter = 0
        self.current_weight_patches_uuid = None
        self.model_offload_buffer_memory = 0
        self._size = size_bytes


class _Outcome:
    """Distinguishes a reply from the two silent-failure outcomes."""

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return self._name


WORKER_GONE = _Outcome("WORKER_GONE")   # process died; its VRAM died with it
SEND_FAILED = _Outcome("SEND_FAILED")   # worker alive; weights still resident


class SubprocessModelPatcher:
    """Standalone duck-type for a worker-resident model.

    Keyed to a worker generation: a patcher left over from a crashed or
    restarted worker reports itself as already offloaded rather than failing.
    """

    #: Everything ComfyUI reads off ``LoadedModel.model``. Kept in sync by
    #: tests/test_model_patcher_surface.py, which greps ComfyUI for the real
    #: access sites and fails when upstream starts touching something new.
    COMFY_SURFACE = frozenset({
        # attributes
        "load_device", "offload_device", "parent", "model", "clone_base_uuid",
        # methods
        "model_size", "loaded_size", "current_loaded_device", "model_dtype",
        "model_patches_to", "model_patches_models", "partially_load",
        "partially_unload", "detach", "lowvram_patch_counter", "is_dynamic",
        "is_clone", "get_nested_additional_models",
    })

    def __init__(self, worker, worker_generation, model_id, model_size,
                 load_device, offload_device, kind="other"):
        self.load_device = load_device
        self.offload_device = offload_device
        # ComfyUI's LoadedModel._set_model reads .parent; clone chains do not
        # exist for subprocess models, so it is always None.
        self.parent = None
        self.clone_base_uuid = None
        self.size = model_size
        self.model = SubprocessModel(model_size, offload_device)
        self._worker = worker
        self._worker_generation = worker_generation
        self._model_id = model_id
        self._kind = kind  # "unet", "clip", "vae", "other"

    # ------------------------------------------------------------------
    # IPC
    # ------------------------------------------------------------------

    def _worker_gone(self):
        return not self._worker.is_alive()

    def _mark_offloaded(self):
        """Local bookkeeping for 'the weights are not on the GPU any more'."""
        self.model.device = self.offload_device
        self.model.model_loaded_weight_memory = 0
        self.model.model_offload_buffer_memory = 0

    def _send(self, command, *, quiet_on_loss, **kwargs):
        """Send an IPC command. Returns the reply, or a sentinel.

        Eviction paths must not raise -- they run inside ComfyUI's free_memory
        loop -- but the two ways they can fail mean opposite things about the
        card, and must not collapse into one answer:

        WORKER_GONE   the process died, so its VRAM died with it. Genuinely
                      freed; safe to report as offloaded.
        SEND_FAILED   the worker is ALIVE and refused the command. The weights
                      are still resident. Claiming otherwise tells ComfyUI it
                      reclaimed bytes it did not, and zeroes loaded_size so the
                      model is never picked for eviction again -- every later
                      admission decision is then computed against a card
                      believed to have that much more free.
        """
        if self._worker_gone():
            if quiet_on_loss:
                log.warning("Worker for model '%s' is gone; treating as offloaded",
                            self._model_id)
                self._mark_offloaded()
                return WORKER_GONE
            raise RuntimeError(
                f"Subprocess worker died; model '{self._model_id}' is no longer "
                f"available. Please reload the model node.")
        try:
            return self._worker.send_command(command, model_id=self._model_id, **kwargs)
        except Exception as e:
            if quiet_on_loss:
                log.warning("Model '%s': %s failed (%s); worker is alive, so the "
                            "weights are still resident", self._model_id, command, e)
                return SEND_FAILED
            raise

    # ------------------------------------------------------------------
    # The surface ComfyUI actually uses
    # ------------------------------------------------------------------

    def model_size(self):
        return self.size

    def loaded_size(self):
        return self.model.model_loaded_weight_memory

    def current_loaded_device(self):
        return self.model.device

    def model_dtype(self):
        return None  # dtype is the worker's business

    def model_patches_to(self, *args, **kwargs):
        return None  # no local patches to move

    def model_patches_models(self):
        return []

    def get_nested_additional_models(self):
        return []

    def lowvram_patch_counter(self):
        return 0

    def is_dynamic(self):
        # False excludes this object from every dynamic-pin path
        # (dynamic_pins, loaded_ram_size, pinned_memory_size,
        # models_for_pin_eviction) -- where most upstream churn lives.
        return False

    def is_clone(self, other):
        return (isinstance(other, SubprocessModelPatcher)
                and self._worker is other._worker
                and self._model_id == other._model_id)

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        """Load up to ``extra_memory`` more bytes; return bytes actually loaded.

        ComfyUI passes 1e32 to mean "as much as fits"; clamp so the wire carries
        a sane integer. The worker runs the REAL ModelPatcher.partially_load.
        """
        want = int(min(float(extra_memory), float(self.size))) if extra_memory else 0
        if want <= 0:
            want = self.size
        size_mb = self.size // (1024 * 1024)
        _log_vram(f"Before load '{self._model_id}' ({size_mb} MB) -> {device_to}")
        r = self._send("model_partial_load", quiet_on_loss=False,
                       extra_bytes=want, device=str(device_to))
        loaded = int((r or {}).get("loaded", 0))
        resident = int((r or {}).get("resident", loaded))
        self.model.model_loaded_weight_memory = resident
        if resident > 0:
            self.model.device = device_to
        _log_vram(f"After load '{self._model_id}' (+{loaded // (1024 * 1024)} MB)")
        return loaded

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        """Free up to ``memory_to_free`` bytes; return bytes ACTUALLY freed.

        A short return is the designed path, not a failure: ComfyUI compares
        ``freed >= memory_to_free`` and escalates to ``detach()`` itself.
        """
        resident_before = self.model.model_loaded_weight_memory
        if resident_before <= 0:
            return 0
        want = int(memory_to_free) if memory_to_free else resident_before
        _log_vram(f"Before partial offload '{self._model_id}' (-{want // (1024 * 1024)} MB)")
        r = self._send("model_partial_unload", quiet_on_loss=True, bytes_to_free=want)
        if r is WORKER_GONE:
            # The VRAM went with the process; everything resident is freed.
            return resident_before
        if r is SEND_FAILED:
            # Nothing was freed, and loaded_size stays put so ComfyUI keeps
            # escalating instead of believing the bytes came back.
            return 0
        freed = int(r.get("freed", 0))
        resident = int(r.get("resident", max(0, resident_before - freed)))
        self.model.model_loaded_weight_memory = resident
        if resident <= 0:
            self.model.device = self.offload_device
        _log_vram(f"After partial offload '{self._model_id}' (-{freed // (1024 * 1024)} MB)")
        return freed

    def detach(self, unpatch_all=True):
        """Full unload. Idempotent, and never raises -- runs inside free_memory."""
        if self.model.model_loaded_weight_memory <= 0 and \
                self.model.device == self.offload_device:
            return self.model  # already off the GPU; skip the round trip
        size_mb = self.size // (1024 * 1024)
        _log_vram(f"Before offload '{self._model_id}' ({size_mb} MB)")
        r = self._send("model_to_device", quiet_on_loss=True,
                       device=str(self.offload_device))
        if r is SEND_FAILED:
            # Still on the card. Report nothing reclaimed rather than logging
            # an offload that did not happen.
            _log_vram(f"Offload '{self._model_id}' FAILED; still resident")
            return self.model
        self._mark_offloaded()
        _log_vram(f"After offload '{self._model_id}' ({size_mb} MB)")
        return self.model

    # ------------------------------------------------------------------
    # Explicitly unsupported (reachable only if a pack ever returns a MODEL)
    # ------------------------------------------------------------------

    def add_patches(self, *args, **kwargs):
        raise NotImplementedError(
            f"Model '{self._model_id}' lives in a subprocess and cannot be "
            f"patched in-process. Weight patches (LoRA etc.) must be applied "
            f"inside the pack's own environment, where a real ModelPatcher "
            f"holds the weights. Silently storing patches that never apply "
            f"would produce wrong output with no error.")

    add_hook_patches = add_patches

    def clone(self, *args, **kwargs):
        raise NotImplementedError(
            f"Model '{self._model_id}' is a subprocess proxy and cannot be "
            f"cloned: a clone would share a worker-side model while ComfyUI "
            f"treated the copies as independent. Clone inside the pack instead.")

    def __getattr__(self, name):
        # Only reached when normal lookup fails.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)  # let copy/pickle/weakref protocols probe
        raise AttributeError(
            f"SubprocessModelPatcher has no {name!r}. ComfyUI touched a "
            f"ModelPatcher member this proxy does not implement, which means "
            f"the upstream contract changed. Implement it here and add it to "
            f"COMFY_SURFACE -- do not inherit ModelPatcher to make it go away.")
