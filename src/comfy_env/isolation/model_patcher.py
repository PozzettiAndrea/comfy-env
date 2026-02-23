"""Proxy ModelPatcher for subprocess-isolated models.

SubprocessModelPatcher registers subprocess GPU models with ComfyUI's memory
manager so they participate in VRAM eviction. When ComfyUI needs VRAM, it
calls unpatch_model() which sends an IPC command to move the model to CPU.
When the model is needed again, patch_model() sends it back to GPU.

Models are auto-detected: the worker hooks Module.to() and .cuda() to catch
any nn.Module that lands on CUDA. Zero per-repo changes needed.
"""

import logging

import comfy.model_patcher
import comfy.model_management

log = logging.getLogger("comfy_env.model_patcher")


class SubprocessModel:
    """Lightweight proxy standing in for ModelPatcher's .model attribute.

    Tracks device/memory state without holding actual weights -- the real
    weights live in the subprocess.  ComfyUI's ModelPatcher constructor and
    LoadedModel read several attributes off .model; we provide them all here.
    """

    def __init__(self, size_bytes, device):
        self.device = device
        self.model_loaded_weight_memory = 0
        self.model_lowvram = False
        self.lowvram_patch_counter = 0
        self.current_weight_patches_uuid = None
        self.model_offload_buffer_memory = 0
        self._size = size_bytes

    def to(self, *args, **kwargs):
        # No-op -- actual device move happens via IPC
        return self

    def state_dict(self):
        return {}  # No local weights

    def modules(self):
        return iter([])  # No submodules to iterate

    def parameters(self):
        return iter([])  # No local parameters

    def __class_name__(self):
        return "SubprocessModel"


class SubprocessModelPatcher(comfy.model_patcher.ModelPatcher):
    """ModelPatcher that proxies device moves to a subprocess worker via IPC.

    The patcher is keyed to a specific worker generation so that stale patchers
    (from a crashed/restarted worker) raise a clean error instead of silently
    failing.
    """

    def __init__(self, worker, worker_generation, model_id, model_size,
                 load_device, offload_device, kind="other"):
        proxy_model = SubprocessModel(model_size, offload_device)
        super().__init__(proxy_model, load_device, offload_device, size=model_size)
        self._worker = worker
        self._worker_generation = worker_generation
        self._model_id = model_id
        self._kind = kind  # "unet", "clip", "vae", "other"

    def _check_worker(self):
        """Raise if the worker has been replaced since this patcher was created."""
        if not self._worker.is_alive():
            raise RuntimeError(
                f"Subprocess worker died; model '{self._model_id}' is no longer available. "
                f"Please reload the model node."
            )

    def _send_device_command(self, device_str):
        """Send model_to_device IPC command."""
        self._check_worker()
        try:
            self._worker.send_command(
                "model_to_device",
                model_id=self._model_id,
                device=device_str,
            )
        except RuntimeError as e:
            if "not registered" in str(e):
                # Worker was restarted — model no longer exists in subprocess.
                # For offload: model is already gone from VRAM, just update state.
                log.warning("Model '%s' gone from worker (restarted?), treating as offloaded",
                            self._model_id)
                self.model.device = self.offload_device
                self.model.model_loaded_weight_memory = 0
                self.model.model_offload_buffer_memory = 0
                return
            raise

    # -- ModelPatcher overrides --

    def patch_model(self, device_to=None, lowvram_model_memory=0,
                    load_weights=True, force_patch_weights=False):
        """Load model to GPU in subprocess."""
        device_to = device_to or self.load_device
        self._send_device_command(str(device_to))
        self.model.device = device_to
        self.model.model_loaded_weight_memory = self.size
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        """Offload model to CPU in subprocess."""
        device_to = device_to or self.offload_device
        self._send_device_command(str(device_to))
        self.model.device = device_to
        self.model.model_loaded_weight_memory = 0
        self.model.model_offload_buffer_memory = 0

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        """Full load (no partial loading for subprocess models)."""
        self.patch_model(device_to, load_weights=True)
        return self.size

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        """Full unload."""
        self.unpatch_model(device_to)
        return self.size

    def partially_unload_ram(self, ram_to_unload):
        pass

    def detach(self, unpatch_all=True):
        self.unpatch_model(self.offload_device)
        return self.model

    def clone(self):
        n = SubprocessModelPatcher(
            self._worker, self._worker_generation, self._model_id,
            self.size, self.load_device, self.offload_device, self._kind,
        )
        n.model.device = self.model.device
        n.model.model_loaded_weight_memory = self.model.model_loaded_weight_memory
        return n

    def cleanup(self):
        pass  # Worker stays alive

    def model_patches_to(self, *args, **kwargs):
        pass  # No local patches to move

    def model_patches_models(self):
        return []  # No sub-models

    def is_clone(self, other):
        if isinstance(other, SubprocessModelPatcher):
            return (self._worker is other._worker
                    and self._model_id == other._model_id)
        return False

    def current_loaded_device(self):
        return self.model.device

    def model_dtype(self):
        return None  # Subprocess handles dtype internally

    def model_state_dict(self, filter_prefix=None):
        return {}

    def get_nested_additional_models(self):
        return []

    def inject_model(self):
        pass

    def eject_model(self):
        pass

    def pre_run(self):
        pass

    def is_dynamic(self):
        return False

    def get_all_callbacks(self, *args, **kwargs):
        return []

    def get_ram_usage(self):
        return self.model_size()
