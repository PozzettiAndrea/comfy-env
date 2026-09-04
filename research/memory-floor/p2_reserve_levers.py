"""P2: which levers actually change what the host does with VRAM?

The floor rests on this. comfy-env stops re-deriving ComfyUI's arithmetic
and instead uses ComfyUI's own controls. There are two candidates, and they
do not both work on both paths:

  preventive  publish a reserve so the host never takes the memory. On the
              legacy path that is comfy.model_management.EXTRA_RESERVED_VRAM,
              read live by extra_reserved_memory() on every load.
  reactive    ask the host to give back memory it already took, by calling
              comfy.model_management.free_memory, which is public and evicts
              the host's own models unconditionally.

Measured conclusion (RTX 3090, ComfyUI 0.33.0, comfy-aimdo 0.4.13):

  legacy  preventive WORKS, reactive works.
  paged   preventive is INERT and cannot be made to work at runtime, so the
          floor on this path must be reactive.

Why preventive fails under aimdo: ModelPatcherDynamic ignores
lowvram_model_memory and decides residency at page-fault time, so the
ComfyUI reserve never enters the decision. aimdo's own headroom is fixed
when its devices are initialised: the global `lib.set_simple_vram_headroom`
is inert once running (measured with NVML pressure both on and off),
`init_devices` returns False on a second call, and calling `control.init`
again SEGFAULTS the process. So on an aimdo host the reserve is whatever
--reserve-vram set at launch, and comfy-env cannot move it afterwards.

Both experiments must create genuine pressure or they measure nothing. Two
earlier versions of this file reported "inert" for a working lever purely
because the model still fitted, and a third reported "does not evict" while
asking for less memory than was already free. Sizes derive from measured
free VRAM for that reason.
"""

import gc
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import harness as H  # noqa: E402

H.bootstrap()

MODEL_GIB = 6.0
LAYER_N = 8192


def _reserve_bytes(free_bytes, model_bytes):
    """A reserve big enough that the model can no longer fully fit."""
    return max(0, int(free_bytes - model_bytes * 0.5))


def _aimdo_usage():
    """What aimdo says it holds. torch cannot see aimdo memory at all, so on
    the paged path this is the only honest metric."""
    try:
        import comfy_aimdo.control as control
        return int(control.get_total_vram_usage())
    except Exception:
        return None


def _load(mm, mp, gib, fault_in=False):
    import torch
    import torch.nn as nn
    layers = max(1, int(gib * H.GIB / (LAYER_N * LAYER_N * 2)))
    model = nn.Sequential(*[
        nn.Linear(LAYER_N, LAYER_N, bias=False) for _ in range(layers)
    ]).half()
    patcher = mp.CoreModelPatcher(
        model, load_device=mm.get_torch_device(),
        offload_device=torch.device("cpu"))
    mm.load_models_gpu([patcher])
    if fault_in:
        # Under aimdo nothing is resident until a page faults.
        with torch.no_grad():
            x = torch.randn(1, LAYER_N, dtype=torch.float16, device="cuda")
            try:
                patcher.model(x)
            except Exception:
                pass
        torch.cuda.synchronize()
    return patcher


def _drop(mm, patcher):
    import torch
    mm.unload_all_models()
    del patcher
    gc.collect()
    torch.cuda.empty_cache()


def leg_legacy_preventive(r):
    import comfy.model_management as mm
    import comfy.model_patcher as mp

    base = int(mm.EXTRA_RESERVED_VRAM)
    try:
        p = _load(mm, mp, MODEL_GIB)
        off, size = int(p.loaded_size()), int(p.model_size())
        _drop(mm, p)

        reserve = _reserve_bytes(
            mm.get_free_memory(mm.get_torch_device()), size)
        mm.EXTRA_RESERVED_VRAM = base + reserve
        r.check("P2.1 the reserve is read live, on every call",
                mm.extra_reserved_memory() == mm.EXTRA_RESERVED_VRAM,
                H.gib(mm.extra_reserved_memory()))

        p = _load(mm, mp, MODEL_GIB)
        on = int(p.loaded_size())
        _drop(mm, p)
        r.check("P2.2 legacy preventive: the reserve reduces host residency",
                on < off,
                "{} -> {} of a {} model, reserve {}".format(
                    H.gib(off), H.gib(on), H.gib(size), H.gib(reserve)))
    finally:
        mm.EXTRA_RESERVED_VRAM = base


def leg_paged(r):
    import os
    import comfy.memory_management as cmm
    import comfy.model_management as mm
    import comfy.model_patcher as mp
    from comfy_env import memory_manager as memmgr

    os.environ["COMFY_ENV_WORKER_AIMDO"] = "1"
    if not memmgr.maybe_enable_aimdo(log=lambda m: None):
        r.note("aimdo will not enable here; paged legs skipped")
        return
    r.check("P2.3 aimdo is the active manager for these legs",
            getattr(cmm, "aimdo_enabled", False) is True)

    import comfy_aimdo.control as control

    base = int(mm.EXTRA_RESERVED_VRAM)
    try:
        p = _load(mm, mp, MODEL_GIB, fault_in=True)
        off_aimdo = _aimdo_usage()
        off_ledger = int(p.loaded_size())
        size = int(p.model_size())
        _drop(mm, p)
        r.check("P2.4 aimdo's own accounting is the metric on this path",
                off_aimdo is not None and off_aimdo > 0,
                "aimdo={} ComfyUI ledger={}".format(
                    H.gib(off_aimdo), H.gib(off_ledger)))

        reserve = _reserve_bytes(
            mm.get_free_memory(mm.get_torch_device()), size)

        # (a) the ComfyUI reserve, which the first design assumed would work
        mm.EXTRA_RESERVED_VRAM = base + reserve
        p = _load(mm, mp, MODEL_GIB, fault_in=True)
        comfy_only = _aimdo_usage()
        _drop(mm, p)
        mm.EXTRA_RESERVED_VRAM = base

        # (b) aimdo's own global headroom setter, at runtime
        setter = getattr(control.lib, "set_simple_vram_headroom", None)
        lever = None
        if setter is not None:
            setter(reserve)
            p = _load(mm, mp, MODEL_GIB, fault_in=True)
            lever = _aimdo_usage()
            _drop(mm, p)
            setter(0)

        r.check("P2.5 paged preventive is inert, by BOTH levers",
                (comfy_only or 0) >= (off_aimdo or 0) * 0.9
                and (lever is None or lever >= (off_aimdo or 0) * 0.9),
                "baseline={} comfy_reserve={} aimdo_headroom={} (reserve {})"
                .format(H.gib(off_aimdo), H.gib(comfy_only), H.gib(lever),
                        H.gib(reserve)))
        r.note("hence the floor is reactive on this path: aimdo's headroom is "
               "fixed at init_devices and cannot be moved afterwards")

        # (c) the reactive path, which is what the floor uses here
        p = _load(mm, mp, MODEL_GIB, fault_in=True)
        before = _aimdo_usage()
        dev = mm.get_torch_device()
        free_now = mm.get_free_memory(dev)
        # Ask for MORE than is free, or ComfyUI correctly does nothing: its
        # eviction loop breaks as soon as free memory exceeds the target.
        mm.free_memory(int(free_now + 4 * H.GIB), dev)
        after = _aimdo_usage()
        _drop(mm, p)
        r.check("P2.6 paged reactive: asking the host to free DOES evict",
                (after or 0) < (before or 1) * 0.5,
                "{} -> {}".format(H.gib(before), H.gib(after)))
    finally:
        mm.EXTRA_RESERVED_VRAM = base


def main():
    r = H.Report("P2 which VRAM levers actually work, per path")
    if not H.gpu_is_idle():
        r.note("WARNING: another process holds the GPU; figures polluted")
    r.note("free before: {}".format(H.gib(H.smi_free_bytes())))
    leg_legacy_preventive(r)
    leg_paged(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
