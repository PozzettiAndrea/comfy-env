"""P7: does the admission time ask really take VRAM back from an idle worker?

The unit tests pin the planner and the wiring. This drives the real thing:
two real workers, a real 6 GiB paged model resident in one, a real budget
request from the other through the parent's own callback, and nvidia-smi as
the witness. The property under test is the one that closes the largest gap
against native ComfyUI: the host, having evicted everything it owns, gets
the rest from a process ComfyUI cannot reach.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

H.bootstrap()
GIB = H.GIB

import comfy.model_management as mm  # noqa: E402,F401  (bootstraps the host manager)
from comfy_env.isolation import pool  # noqa: E402


def main():
    r = H.Report("P7 admission time ask takes VRAM back from an idle worker")
    # The fleet's default on NVIDIA is the pager; PAGED=0 measures the
    # legacy ledger path instead, which costs a real copy to host RAM.
    env = {} if os.environ.get("PAGED") == "0" else {"COMFY_ENV_WORKER_AIMDO": "1"}
    holder = H.spawn("holder", extra_env=env)
    requester = H.spawn("requester", extra_env=env)
    try:
        for w in (holder, requester):
            w.register_callback("request_vram_budget", pool._handle_vram_budget)
            w.call_module("probe", "manager_state")

        pool._WORKER_POOL.clear()
        pool._WORKER_POOL["holder"] = (holder, 1)
        pool._WORKER_POOL["requester"] = (requester, 1)

        r.check("worker advertises partial_release",
                bool(getattr(holder, "supports_partial_release", False)))

        st = holder.call_module("probe", "manager_state")
        r.note("holder manager: {}".format(
            {k: st.get(k) for k in ("patcher", "aimdo_enabled", "dynamic",
                                    "manager", "core_patcher") if k in st}))
        loaded = holder.call_module("probe", "load", gib=6.0, ops=True)
        r.note("model: {}".format(loaded))
        holder.call_module("probe", "forward", steps=1)
        held = holder.call_module("probe", "vram_truth")
        resident = max(int(held.get("aimdo_total", 0)),
                       int(held.get("torch_reserved", 0)))
        pool._WORKER_HELD["holder"] = resident
        free_before = H.smi_free_bytes() or 0
        r.note("holder resident {}, device free {}".format(
            H.gib(resident), H.gib(free_before)))
        r.check("the holder really holds a big model", resident > 4 * GIB)

        # The ask, exactly as admission calls it: the host has already given
        # up what it owns and is still short by this much.
        shortfall = 4 * GIB
        t0 = time.perf_counter()
        _receipts = []
        _orig = holder.send_command_no_spawn

        def _spy(method, **kw):
            out = _orig(method, **kw)
            if method == "partial_release":
                _receipts.append((out or {}).get("receipt"))
            return out
        holder.send_command_no_spawn = _spy
        freed = pool._ask_idle_workers(shortfall, requester_key="requester")
        r.note("receipt: {}".format(_receipts[0] if _receipts else None))
        elapsed = time.perf_counter() - t0
        time.sleep(0.5)
        free_after = H.smi_free_bytes() or 0
        after = holder.call_module("probe", "vram_truth")
        resident_after = max(int(after.get("aimdo_total", 0)),
                             int(after.get("torch_reserved", 0)))

        r.note("asked {} -> receipt {} in {:.2f}s; holder {} -> {}; "
               "device free {} -> {}".format(
                   H.gib(shortfall), H.gib(freed), elapsed,
                   H.gib(resident), H.gib(resident_after),
                   H.gib(free_before), H.gib(free_after)))
        r.check("the holder gave memory back", resident_after < resident * 0.75)
        r.check("the card really has more room",
                free_after - free_before > shortfall * 0.7)
        r.check("the receipt is not a lie", freed > shortfall * 0.7)
        r.check("our copy of what it holds followed the receipt down",
                pool._WORKER_HELD["holder"] < resident)

        # And the model is still loaded: this is what makes an ask cheaper
        # than an eviction, so the next call refaults instead of reloading.
        t1 = time.perf_counter()
        # What a real node call does: back through ComfyUI's own admission,
        # which refaults a paged model and re-copies a legacy one.
        holder.call_module("probe", "reload")
        holder.call_module("probe", "forward", steps=1)
        refault = time.perf_counter() - t1
        back = holder.call_module("probe", "vram_truth")
        resident_back = max(int(back.get("aimdo_total", 0)),
                            int(back.get("torch_reserved", 0)))
        r.note("refault after the ask: {:.2f}s, back to {}".format(
            refault, H.gib(resident_back)))
        r.check("the model was kept, not unloaded", resident_back > resident * 0.75)
        r.finish()
    finally:
        for w in (holder, requester):
            try:
                w.call_module("probe", "release_all")
            except Exception:
                pass
            try:
                w.shutdown()
            except Exception:
                pass
        pool._WORKER_POOL.clear()
        pool._WORKER_HELD.clear()
    return r


if __name__ == "__main__":
    main()
    # Hard exit: the transport's CUDA IPC handles outlive the interpreter's
    # teardown and torch aborts on the unlink. Not a finding, just teardown.
    os._exit(0)
