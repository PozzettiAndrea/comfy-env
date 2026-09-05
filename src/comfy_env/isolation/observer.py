"""An optional read-only listener in ComfyUI's loaded-model list.

Off by default. comfy-env's floor needs nothing from ComfyUI's internals
beyond values it reads and one number it publishes, and that is the shape
worth defaulting to. But two signals only exist inside that list, and some
operators will want them:

* The Free-memory button. It reaches ComfyUI's eviction loop as a request to
  free 1e30 bytes, a value nothing else ever asks for, so an entry in the
  list learns the user pressed it. Without this, the built-in button frees
  the host's models and silently leaves worker memory alone.
* Host memory pressure. Being asked to free at all is the only in-process
  notice that ComfyUI is short of VRAM. Idle release cannot cover that case,
  because during it the workers are not idle.

Why this is safe to duck type when the old model proxy was not: the proxy
claimed to HOLD memory, so ComfyUI relied on its numbers and called deeper
and deeper into it, and both of comfy-env's historical loud breaks were new
attribute reads landing on it. This object claims to hold nothing. ComfyUI
asks, gets zero, and moves on to something it can actually evict. Nothing
downstream depends on an answer, so an unknown attribute can honestly return
a harmless default instead of a lie.

It is still a coupling, which is why it is a switch and why the switch is
off. Set COMFY_ENV_MEMORY_OBSERVER=on to accept it.
"""

ENV_VAR = "COMFY_ENV_MEMORY_OBSERVER"


class _Detachable:
    """Stands in for LoadedModel.model_finalizer, which ComfyUI detaches when
    it drops an entry. A no-op, but it must exist and be callable."""

    def detach(self):
        pass

#: ComfyUI's unload_all_models asks its eviction loop for this many bytes.
#: Nothing else ever asks for anything near it, so it identifies the Free
#: memory button rather than ordinary pressure.
FREE_ALL_SENTINEL = 1e30

#: Anything above this is the sentinel rather than a real request. A real
#: request is bounded by the size of a model plus headroom; this is orders
#: of magnitude above any card.
SENTINEL_FLOOR = 1e29


def is_free_all(memory_required, smart_memory=True) -> bool:
    """Whether an eviction request is the Free-memory button.

    Catches the wrong implementation of comparing against 1e30 exactly:
    upstream computes the value, and float arithmetic on the way in has
    already changed it once.

    ``smart_memory`` is ComfyUI's own switch inverted: under
    --disable-smart-memory every request is 1e32 (model_management.py), so
    the sentinel stops identifying the button and starts identifying every
    load. There the honest answer is that ALL of them mean release, because
    that is what the flag means, so the caller gets True and the pressure
    callback never fires; the alternative, reading ordinary pressure as the
    button, would broadcast a full release on every host load.
    """
    try:
        big = float(memory_required) >= SENTINEL_FLOOR
    except (TypeError, ValueError):
        return False
    return big if smart_memory else True


def enabled(env) -> bool:
    """Off unless explicitly turned on. Unset means off, not 'default on'."""
    value = (env or {}).get(ENV_VAR)
    return str(value or "").strip().lower() in ("1", "true", "on", "yes")


#: A private identity for the clone check. ComfyUI compares
#: `.model.clone_base_uuid` against a real model's uuid to decide what to
#: keep loaded (unload_model_and_clones). Anything that never equals a uuid
#: keeps the observer in the keep list; None does NOT, because a target
#: whose own uuid is None would then match. Deliberate, not incidental.
_NOT_A_CLONE = object()


class MemoryObserver:
    """Looks enough like a loaded model to be asked, and holds nothing.

    Every method answers "I have no memory for you", which is true: worker
    memory lives in another process and this object is not a handle to it.
    The value is entirely in being CALLED.

    The named attributes below are the ones ComfyUI reads on an entry, and
    each is defined rather than left to __getattr__ because the catch-all
    returns a callable, and a callable is the wrong answer for every one of
    them: `real_model()` answering None makes cleanup_models POP this entry
    the first time any host model is collected, which is exactly when the
    Free button or an OOM has just fired.
    """

    def __init__(self, device=None, on_free_all=None, on_pressure=None):
        self.device = device
        self._on_free_all = on_free_all
        self._on_pressure = on_pressure
        self.model = self          # ComfyUI reaches through .model
        # cleanup_models (model_management.py) prunes every entry whose
        # real_model() is None. Answer with something that outlives the
        # entry: self, held by the list that holds us.
        self.real_model = lambda: self
        self.model_finalizer = _Detachable()
        self.clone_base_uuid = _NOT_A_CLONE
        self.load_device = device
        self.offload_device = device
        self.parent = None

    @property
    def currently_used(self):
        """Always False, and read only.

        loaded_models(only_currently_used=True) hands entries straight back
        to load_models_gpu (controlnet.py and several comfy_extras nodes),
        which would then run the whole admission path against this object.
        free_memory writes False on every pass and load_models_gpu writes
        True on an __eq__ match, so this is a property rather than an
        attribute: no future write can flip it.
        """
        return False

    @currently_used.setter
    def currently_used(self, _value):
        pass

    def clone(self, *args, **kwargs):
        """multigpu.py calls .clone() on entries it reads from the list. The
        catch-all would hand back None and the caller would use it."""
        raise RuntimeError(
            "comfy-env's memory observer is not a model and cannot be cloned")

    def is_clone(self, other=None):
        return False

    def model_patches_models(self):
        return []

    # -- what ComfyUI's eviction loop asks -------------------------------
    def is_dead(self):
        return False

    def model_memory(self):
        return 0

    def model_offloaded_memory(self):
        return 0

    def model_loaded_memory(self):
        return 0

    def loaded_size(self):
        return 0

    def model_size(self):
        return 0

    def is_dynamic(self):
        return False

    def model_unload(self, memory_to_free=None, *args, **kwargs):
        """Asked to free. Report, then decline honestly.

        Returning False means "I did not unload", which is true and leaves
        ComfyUI's loop to try something that can. The signal is the call.

        The callbacks run on ComfyUI's own thread, inside free_memory,
        inside a node: they must post and return, never wait on a worker.
        A blocked callback here stalls every host load.
        """
        try:
            smart = True
            try:
                import comfy.model_management as mm
                smart = not getattr(mm, "DISABLE_SMART_MEMORY", False)
            except Exception:
                pass
            if is_free_all(memory_to_free, smart_memory=smart):
                if self._on_free_all is not None:
                    self._on_free_all()
            elif memory_to_free and self._on_pressure is not None:
                self._on_pressure(int(memory_to_free))
        except Exception:
            pass
        return False

    def __getattr__(self, name):
        """Anything not named above answers harmlessly.

        Legitimate here and NOT legitimate on a model proxy: this object
        reports holding nothing, so no caller depends on the answer. On the
        proxy the same permissiveness produced confident wrong numbers.
        """
        if name.startswith("__"):
            raise AttributeError(name)

        def _nothing(*args, **kwargs):
            return None

        return _nothing
