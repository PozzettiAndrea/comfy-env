"""P1: a comfy-aimdo patch bump must not strand a worker on the ledger.

Question: when the host's comfy-aimdo differs from the worker's only in the
patch version, does the worker still enable aimdo?

Before the protocol-level guard the answer was no, and it was live on two of
nineteen environments on this machine. This proves the fix on real wheels.
"""

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import harness as H  # noqa: E402

H.bootstrap()


def main():
    r = H.Report("P1 comfy-aimdo skew is judged on protocol, not version")

    from comfy_env.memory_manager import (
        LEVEL_ENV_VAR, VERSION_ENV_VAR, aimdo_installed_level, aimdo_version,
    )

    # What this machine's host actually has.
    try:
        import comfy_aimdo.control as ctl
        host_level = aimdo_installed_level(ctl)
        host_version = aimdo_version()
    except ImportError:
        host_level, host_version = None, None
    r.note("host aimdo: version={} level={}".format(host_version, host_level))

    # 1. The worker environment's own wheel reports a level.
    worker = H.spawn("p1-level", {"COMFY_ENV_WORKER_AIMDO": "1"})
    try:
        lvl = worker.call_module("probe", "aimdo_level")
        r.check("P1.1 worker reports a protocol level from its own wheel",
                isinstance(lvl.get("level"), int) and lvl["level"] >= 1,
                "level={} version={}".format(lvl.get("level"), lvl.get("version")))
        worker_level = lvl.get("level")
        worker_version = lvl.get("version")
    finally:
        worker.shutdown()

    if worker_level is None:
        r.note("no aimdo in the worker env; the rest cannot run")
        return r.finish()

    # 2. THE REGRESSION: a parent one patch ahead, same protocol level.
    #    The old guard compared version strings and refused here.
    faked = "{}.{}.{}".format(*(
        [int(x) for x in (worker_version or "0.0.0").split(".")[:2]]
        + [int((worker_version or "0.0.0").split(".")[2]) + 2]
    ))
    worker = H.spawn("p1-skew", {
        "COMFY_ENV_WORKER_AIMDO": "1",
        VERSION_ENV_VAR: faked,
        LEVEL_ENV_VAR: str(worker_level),
    })
    try:
        state = worker.call_module("probe", "manager_state")
        r.check("P1.2 same level, newer parent patch: aimdo still enabled",
                state.get("manager") == "aimdo",
                "manager={} reason={}".format(
                    state.get("manager"), state.get("reason")))
        r.note("worker {} vs parent {}".format(worker_version, faked))
    finally:
        worker.shutdown()

    # 3. A genuine protocol difference must still refuse.
    worker = H.spawn("p1-cross", {
        "COMFY_ENV_WORKER_AIMDO": "1",
        VERSION_ENV_VAR: "9.9.9",
        LEVEL_ENV_VAR: str(worker_level + 1),
    })
    try:
        state = worker.call_module("probe", "manager_state")
        r.check("P1.3 a real protocol difference is refused, loudly",
                state.get("manager") == "ledger"
                and "protocol skew" in (state.get("reason") or ""),
                "manager={} reason={}".format(
                    state.get("manager"), state.get("reason")))
    finally:
        worker.shutdown()

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
