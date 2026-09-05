"""P8: does the Free button reach a worker when the observer is on?

comfy-env patches nothing, so ComfyUI's unload_all_models never leaves the
host process. The optional listener is the whole mechanism: an entry in
current_loaded_models that is asked to free, reports holding nothing, and
posts the signal to the pool. This drives the real chain, host side, with a
real worker holding a real model and nvidia-smi as the witness.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

H.bootstrap()
GIB = H.GIB
os.environ["COMFY_ENV_MEMORY_OBSERVER"] = "1"

import comfy.model_management as mm  # noqa: E402
from comfy_env.isolation import pool  # noqa: E402


def main():
    r = H.Report("P8 the Free button reaches a worker through the observer")
    worker = H.spawn("holder", extra_env={"COMFY_ENV_WORKER_AIMDO": "1"})
    try:
        worker.register_callback("request_vram_budget", pool._handle_vram_budget)
        worker.call_module("probe", "manager_state")
        pool._WORKER_POOL.clear()
        pool._WORKER_POOL["holder"] = (worker, 1)

        r.check("observer installed", pool._install_observer())
        r.check("it is in ComfyUI's own list",
                pool._OBSERVER in mm.current_loaded_models)

        worker.call_module("probe", "load", gib=6.0, ops=True)
        worker.call_module("probe", "forward", steps=1)
        held = worker.call_module("probe", "vram_truth")
        resident = max(int(held.get("aimdo_total", 0)),
                       int(held.get("torch_reserved", 0)))
        free_before = H.smi_free_bytes() or 0
        r.note("worker resident {}, device free {}".format(
            H.gib(resident), H.gib(free_before)))

        # ComfyUI's own Free button path, called exactly as the server does.
        t0 = time.perf_counter()
        mm.unload_all_models()
        # The callback posts to a thread so it can never stall a host load;
        # a user pressing Free waits, we wait too.
        deadline = time.time() + 20
        while time.time() < deadline:
            now = H.smi_free_bytes() or 0
            if now - free_before > resident * 0.5:
                break
            time.sleep(0.5)
        elapsed = time.perf_counter() - t0
        free_after = H.smi_free_bytes() or 0
        after = worker.call_module("probe", "vram_truth")
        resident_after = max(int(after.get("aimdo_total", 0)),
                             int(after.get("torch_reserved", 0)))
        r.note("after Free ({:.1f}s): worker {} -> {}, device free {} -> {}".format(
            elapsed, H.gib(resident), H.gib(resident_after),
            H.gib(free_before), H.gib(free_after)))
        r.check("the worker let go", resident_after < resident * 0.5)
        r.check("the card really got it back",
                free_after - free_before > resident * 0.5)
        r.check("the observer survived being asked",
                pool._OBSERVER in mm.current_loaded_models)

        # The prune that used to unplant it: cleanup_models runs on every
        # host model collection, and the listener answered None to
        # real_model() until today.
        mm.cleanup_models()
        r.check("and survives cleanup_models",
                pool._OBSERVER in mm.current_loaded_models)
        r.finish()
    finally:
        try:
            worker.shutdown()
        except Exception:
            pass
        pool._WORKER_POOL.clear()
    return r


if __name__ == "__main__":
    main()
    os._exit(0)
