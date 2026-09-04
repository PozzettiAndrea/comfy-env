"""P4: the number the host reserves against is physical truth.

The reserve is only as good as the residency it subtracts. Four candidate
sources disagree, and three of them are wrong in the configuration that
matters most.
"""

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import harness as H  # noqa: E402

H.bootstrap()


def main():
    r = H.Report("P4 what a worker holds, measured four ways")
    worker = H.spawn("p4", {"COMFY_ENV_WORKER_AIMDO": "1"})
    cb, _seen = H.budget_callback()
    worker.register_callback("request_vram_budget", cb)
    try:
        info = worker.call_module("probe", "load", gib=4.0, timeout=600)
        r.note("loaded {} of {} ({})".format(
            H.gib(info["loaded"]), H.gib(info["size"]),
            "dynamic" if info["dynamic"] else "legacy"))
        truth = worker.call_module("probe", "forward", steps=1, timeout=600)
        r.note("torch reserved      {}".format(H.gib(truth["torch_reserved"])))
        r.note("comfy ledger        {}".format(H.gib(truth.get("comfy_ledger"))))
        r.note("patcher loaded_size {}".format(H.gib(truth.get("patcher_loaded"))))
        r.note("aimdo accounting    {}".format(H.gib(truth.get("aimdo_total"))))

        report = getattr(worker, "_last_vram_report", None) or {}
        held = report.get("held")
        r.check("P4.1 the worker reports a single measured held scalar",
                isinstance(held, int) and held > 0, H.gib(held))

        biggest = max(int(truth.get("aimdo_total") or 0),
                      int(truth.get("patcher_loaded") or 0),
                      int(truth["torch_reserved"]))
        r.check("P4.2 it is not smaller than any single source",
                (held or 0) >= biggest * 0.95,
                "held={} vs largest source={}".format(
                    H.gib(held), H.gib(biggest)))
        r.check("P4.3 it is within a factor of the real model size",
                (held or 0) >= int(info["size"]) * 0.5,
                "a ledger-only reading is what this catches")
    finally:
        worker.shutdown()
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
