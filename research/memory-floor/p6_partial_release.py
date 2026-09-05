"""P6 (experiment E2): what a partial release costs on the paged path.

The admission time ask (build order item 6) sends an idle worker
``free_memory(shortfall)``. Two numbers decide its shape: how long that takes
for a VBAR backed model, and whether the freed bytes are visible to another
process's driver reading as soon as the call returns. Both measured here,
in process, on a comfy.ops model so the pages really live in a VBAR (a plain
nn.Linear model is moved with weight.to(device) and never pages; that was the
P2 false negative).

Run with the worker interpreter and COMFY_DIR pointing at a ComfyUI tree.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

H.bootstrap()
MODEL_GIB = float(os.environ.get("MODEL_GIB", "6.0"))
LAYER_N = 8192
MIB = H.MIB

import torch  # noqa: E402
import comfy.memory_management as cmm  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.model_patcher as mp  # noqa: E402
import comfy.ops  # noqa: E402
from comfy_env import memory_manager as memmgr  # noqa: E402

os.environ["COMFY_ENV_WORKER_AIMDO"] = "1"
assert memmgr.maybe_enable_aimdo(log=print), memmgr._ENABLE_ERROR
assert cmm.aimdo_enabled is True
assert mp.CoreModelPatcher is mp.ModelPatcherDynamic
import comfy_aimdo.control as control  # noqa: E402

dev = mm.get_torch_device()


def build():
    import torch.nn as nn
    layers = max(1, int(MODEL_GIB * H.GIB / (LAYER_N * LAYER_N * 2)))
    saved = cmm.aimdo_enabled
    cmm.aimdo_enabled = False
    try:
        mods = [comfy.ops.disable_weight_init.Linear(LAYER_N, LAYER_N, bias=False)
                for _ in range(layers)]
    finally:
        cmm.aimdo_enabled = saved
    model = nn.Sequential(*mods).half()
    for m in model:
        m.weight.data.fill_(0.001)
    return model, layers


def forward(patcher):
    with torch.no_grad():
        x = torch.randn(1, LAYER_N, dtype=torch.float16, device=dev)
        y = patcher.model(x)
    torch.cuda.synchronize()
    return float(y.float().abs().sum())


def state(patcher):
    return {
        "loaded": int(patcher.loaded_size()),
        "aimdo": int(control.get_total_vram_usage()),
        "smi_free": H.smi_free_bytes() or 0,
        "cuda_free": int(torch.cuda.mem_get_info(dev)[0]),
    }


def main():
    r = H.Report("P6 partial release cost on the paged path (E2)")
    # The idle check would see our own context; the caller checks nvidia-smi
    # before starting this script instead.
    model, layers = build()
    patcher = mp.CoreModelPatcher(model, load_device=dev, offload_device=torch.device("cpu"))
    mm.load_models_gpu([patcher])
    forward(patcher)
    full = state(patcher)
    r.note("resident after fault: loaded {} aimdo {} smi_free {}".format(
        H.gib(full["loaded"]), H.gib(full["aimdo"]), H.gib(full["smi_free"])))

    for target_gib in (2, 4, 6):
        before = state(patcher)
        t0 = time.perf_counter()
        mm.free_memory(int(before["cuda_free"] + target_gib * H.GIB), dev)
        elapsed = time.perf_counter() - t0
        after = state(patcher)
        freed = before["loaded"] - after["loaded"]
        smi_gain = after["smi_free"] - before["smi_free"]
        r.note("free {} GiB: took {:.3f}s, loaded {} -> {}, driver free rose {} "
              "(visible at once: {})".format(
                  target_gib, elapsed, H.gib(before["loaded"]), H.gib(after["loaded"]),
                  H.gib(smi_gain), "yes" if smi_gain >= freed * 0.9 else "NO"))
        t1 = time.perf_counter()
        mm.load_models_gpu([patcher])
        forward(patcher)
        refault = time.perf_counter() - t1
        back = state(patcher)
        r.note("  refault: {:.3f}s, loaded back to {}".format(refault, H.gib(back["loaded"])))
        ok = after["loaded"] <= before["loaded"] - target_gib * H.GIB * 0.9 and back["loaded"] >= full["loaded"] * 0.98
        r.check("free {} GiB then refault restores".format(target_gib), ok)
    mm.unload_all_models()
    return r


if __name__ == "__main__":
    rep = main()
    rep.finish()
