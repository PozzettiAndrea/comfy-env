"""Worker-side probe for the memory-floor experiments.

Staged onto the worker's sys.path by ``harness.spawn`` and called through
``worker.call_module("probe", "<func>")``. Runs inside the pack env, so it
may import torch and comfy freely; the parent side must not import this.
"""

import os

_PATCHERS = []


def manager_state():
    """What memory manager this worker resolved to, and why."""
    import sys
    mm = sys.modules.get("memory_manager")
    if mm is None:
        import memory_manager as mm  # staged beside the worker program
    info = dict(mm.describe())
    info["aimdo_level_env"] = os.environ.get("COMFY_ENV_AIMDO_LEVEL")
    info["aimdo_version_env"] = os.environ.get("COMFY_ENV_AIMDO_VERSION")
    return info


def aimdo_level():
    """Protocol level of the comfy-aimdo installed in THIS environment."""
    import sys
    mm = sys.modules.get("memory_manager")
    if mm is None:
        import memory_manager as mm
    try:
        import comfy_aimdo.control as control
    except ImportError as exc:
        return {"error": str(exc)}
    return {
        "level": mm.aimdo_installed_level(control),
        "version": mm.aimdo_version(),
    }


def vram_truth():
    """Every candidate source of 'what this worker holds', in bytes.

    They disagree: torch cannot see aimdo's allocations at all, and
    ComfyUI's own ledger reads zero for a paged model.
    """
    import torch
    out = {
        "torch_reserved": int(torch.cuda.memory_reserved()),
        "torch_allocated": int(torch.cuda.memory_allocated()),
        "pid": os.getpid(),
    }
    try:
        import comfy_aimdo.control as control
        out["aimdo_total"] = int(control.get_total_vram_usage())
    except Exception as exc:
        out["aimdo_total_error"] = str(exc)[:120]
    try:
        out["patcher_loaded"] = int(sum(p.loaded_size() for p in _PATCHERS))
        out["patcher_size"] = int(sum(p.model_size() for p in _PATCHERS))
    except Exception as exc:
        out["patcher_error"] = str(exc)[:120]
    try:
        import comfy.model_management as cmm
        out["comfy_ledger"] = int(sum(
            getattr(p.model, "model_loaded_weight_memory", 0) for p in _PATCHERS))
        out["comfy_total_pinned"] = int(getattr(cmm, "TOTAL_PINNED_MEMORY", 0))
    except Exception as exc:
        out["comfy_ledger_error"] = str(exc)[:120]
    return out


def load(gib=2.0, layer_n=8192, ops=False):
    """Load a model through ComfyUI's own path, so aimdo really pages it.

    ``ops=True`` builds comfy.ops layers, which carry comfy_cast_weights and
    are therefore routed through a VBAR by ModelPatcherDynamic. Plain
    nn.Linear (the default, and what P2 used) takes the weight.to(device)
    branch instead and never pages: that difference is what made P2 read the
    pager's headroom as inert.
    """
    import torch
    import torch.nn as nn
    import comfy.model_management as mm
    import comfy.model_patcher as mp

    per_layer = layer_n * layer_n * 2
    layers = max(1, int(gib * 1024 ** 3 / per_layer))
    if ops:
        import comfy.memory_management as cmm
        import comfy.ops
        saved = cmm.aimdo_enabled
        cmm.aimdo_enabled = False
        try:
            mods = [comfy.ops.disable_weight_init.Linear(layer_n, layer_n, bias=False)
                    for _ in range(layers)]
        finally:
            cmm.aimdo_enabled = saved
    else:
        mods = [nn.Linear(layer_n, layer_n, bias=False) for _ in range(layers)]
    model = nn.Sequential(*mods).half()
    patcher = mp.CoreModelPatcher(
        model, load_device=mm.get_torch_device(),
        offload_device=torch.device("cpu"))
    mm.load_models_gpu([patcher])
    _PATCHERS.append(patcher)
    return {
        "patcher": type(patcher).__name__,
        "dynamic": bool(patcher.is_dynamic()),
        "size": int(patcher.model_size()),
        "loaded": int(patcher.loaded_size()),
    }


def forward(steps=1, layer_n=8192):
    """Fault the model in, which is when a paged worker's memory appears."""
    import torch
    with torch.no_grad():
        for _ in range(steps):
            x = torch.randn(1, layer_n, dtype=torch.float16, device="cuda")
            for p in _PATCHERS:
                x = p.model(x)
    torch.cuda.synchronize()
    return vram_truth()


def reload():
    """What every real node call does before using a model: go back through
    ComfyUI's own admission. After a partial release a paged model refaults
    here and a legacy one is copied back to the device."""
    import comfy.model_management as mm
    mm.load_models_gpu(list(_PATCHERS))
    return vram_truth()


def release_all():
    """Give everything back and report what it cost."""
    import time
    import torch
    import comfy.model_management as mm

    before = vram_truth()
    t0 = time.time()
    mm.unload_all_models()
    _PATCHERS.clear()
    torch.cuda.empty_cache()
    host_empty = getattr(getattr(torch, "_C", None), "_host_emptyCache", None)
    if callable(host_empty):
        host_empty()
    torch.cuda.synchronize()
    return {
        "seconds": round(time.time() - t0, 3),
        "before": before,
        "after": vram_truth(),
    }


def contract_probe():
    """Prove the staged contract module is importable and evaluates the
    worker side entries in this process, with a deliberate miss."""
    import contract as c
    keys = c.required_keys(c.WORKER, (c.FLOOR, c.PAGED))
    ok, failures, notes = c.check(side=c.WORKER, tiers=(c.FLOOR, c.PAGED))
    present = {k: True for k in keys}
    present["comfy_aimdo.model_vbar.vbars_reset_watermark_limits"] = False
    ok2, _f2, notes2 = c.evaluate(present, side=c.WORKER,
                                  tiers=(c.FLOOR, c.PAGED))
    return {"worker_keys": len(keys), "live_ok": ok,
            "live_failures": failures, "live_notes": notes,
            "simulated_miss_ok": ok2, "simulated_miss_notes": notes2}
