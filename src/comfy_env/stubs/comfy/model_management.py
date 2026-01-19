"""
Stub for comfy.model_management in isolated worker processes.

Provides device detection and memory management functions without
requiring the full ComfyUI installation.
"""

import torch


def get_torch_device():
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_free_memory(device=None, torch_free_too=False):
    """Return free VRAM in bytes."""
    if device is None:
        device = get_torch_device()
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        return free
    return 0


def get_total_memory(device=None, torch_total_too=False):
    """Return total VRAM in bytes."""
    if device is None:
        device = get_torch_device()
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        return total
    return 0


def soft_empty_cache(force=False):
    """Clear CUDA cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_all_models():
    """No-op in isolated worker - models managed by the node itself."""
    pass


def interrupt_current_processing(value=True):
    """No-op in isolated worker."""
    pass


def processing_interrupted():
    """Always returns False in isolated worker."""
    return False
