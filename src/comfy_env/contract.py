"""What comfy-env requires of its host, written down and checkable.

comfy-env couples to ComfyUI internals that carry no stability guarantee:
memory management is deliberately outside ComfyUI's versioned API surface,
and it is the fastest moving part of that codebase. Waiting for upstream to
freeze something is not a plan. Writing down what WE depend on is.

Three properties make this useful rather than decorative:

* It is data, not code. Every entry is a literal, so it can be read by a
  static sweep over ComfyUI's history to compute which versions satisfy it,
  instead of that range being asserted by hand and going stale.
* It is checked at startup, on the real tree the operator is running, so a
  missing symbol is announced with a reason instead of discovered later as
  a wrong number or a silent no-op.
* Severity is per entry. Something that would produce WRONG numbers refuses;
  something that merely turns a feature off says so once, by name. That is
  the house rule (degrade on availability, fail loudly on correctness)
  applied to the host surface.

This module imports nothing. It must stay importable in bare CI with no
torch and no comfy, and it is staged into workers, so it must also parse on
the oldest interpreter a worker environment can use.
"""

#: Severity. FATAL entries can produce wrong numbers, wrong eviction, or
#: silent corruption if they are missing, so their absence refuses. DEGRADE
#: entries only switch a feature off, so their absence is one named line.
FATAL = "fatal"
DEGRADE = "degrade"

#: Which tier needs the entry. The floor is always on; the rest map to
#: COMFY_ENV_MEMORY_MANAGEMENT levels.
FLOOR = "floor"
PAGED = "paged"
SHARED = "shared"

#: Which process needs it.
HOST = "host"
WORKER = "worker"
BOTH = "both"

#: One entry per coupling:
#:   module   dotted module name, imported by the checker
#:   attr     attribute comfy-env reads or calls
#:   kind     "attr" (must exist) or "callable" (must exist and be callable)
#:   severity FATAL or DEGRADE
#:   tier     FLOOR, PAGED or SHARED
#:   side     HOST, WORKER or BOTH
#:   why      what breaks without it, in the operator's terms
#:   since    where it came from, when known, so a failure can name a version
CONTRACT = (
    # ---------------------------------------------------------------- floor
    {"module": "comfy.model_management", "attr": "get_torch_device",
     "kind": "callable", "severity": FATAL, "tier": FLOOR, "side": BOTH,
     "why": "no device to account for; every VRAM decision is unanchored",
     "since": None},
    {"module": "comfy.model_management", "attr": "get_free_memory",
     "kind": "callable", "severity": FATAL, "tier": FLOOR, "side": BOTH,
     "why": "cannot tell how much of the card is free, so admission is blind",
     "since": None},
    {"module": "comfy.model_management", "attr": "free_memory",
     "kind": "callable", "severity": FATAL, "tier": FLOOR, "side": HOST,
     "why": "the only way a worker can ask the host to give VRAM back; "
            "without it a pack cannot run beside a host workflow",
     "since": None},
    {"module": "comfy.model_management", "attr": "minimum_inference_memory",
     "kind": "callable", "severity": FATAL, "tier": FLOOR, "side": HOST,
     "why": "the inference reserve upstream books for its own loads; "
            "without it comfy-env under-asks and the worker OOMs",
     "since": None},
    {"module": "comfy.model_management", "attr": "EXTRA_RESERVED_VRAM",
     "kind": "attr", "severity": FATAL, "tier": FLOOR, "side": HOST,
     "why": "the reserve comfy-env publishes so the host stops over-committing",
     "since": "ComfyUI 045377ea, 2024-08-19"},
    {"module": "comfy.model_management", "attr": "extra_reserved_memory",
     "kind": "callable", "severity": FATAL, "tier": FLOOR, "side": HOST,
     "why": "proves the reserve is read live rather than captured at startup",
     "since": "ComfyUI b643eae0, 2024-09-01"},
    {"module": "comfy.model_management", "attr": "vram_state",
     "kind": "attr", "severity": DEGRADE, "tier": FLOOR, "side": HOST,
     "why": "the worker mirrors the host's VRAM mode; without it the worker "
            "picks its own and may disagree",
     "since": None},
    {"module": "comfy.cli_args", "attr": "args",
     "kind": "attr", "severity": DEGRADE, "tier": FLOOR, "side": BOTH,
     "why": "the flag mirror; without it a worker resolves different dtypes "
            "and attention than the host",
     "since": None},

    # ---------------------------------------------------------------- paged
    {"module": "comfy.memory_management", "attr": "aimdo_enabled",
     "kind": "attr", "severity": DEGRADE, "tier": PAGED, "side": BOTH,
     "why": "how a worker learns whether the host pages; absent, host and "
            "worker can silently run different memory managers",
     "since": None},
    {"module": "comfy.model_patcher", "attr": "ModelPatcherDynamic",
     "kind": "attr", "severity": DEGRADE, "tier": PAGED, "side": WORKER,
     "why": "the paging patcher; absent, the worker runs the legacy ledger",
     "since": None},
    {"module": "comfy_aimdo.control", "attr": "init",
     "kind": "callable", "severity": DEGRADE, "tier": PAGED, "side": WORKER,
     "why": "starts the pager; absent, the worker runs the legacy ledger",
     "since": None},
    {"module": "comfy_aimdo.control", "attr": "init_devices",
     "kind": "callable", "severity": DEGRADE, "tier": PAGED, "side": WORKER,
     "why": "per device headroom, which is fixed here and cannot be changed "
            "later; absent, the worker pages without mirroring host reserve",
     "since": "comfy-aimdo 0.4.10 for the (index, bytes) form"},
    {"module": "comfy_aimdo.control", "attr": "get_total_vram_usage",
     "kind": "callable", "severity": FATAL, "tier": PAGED, "side": WORKER,
     "why": "the only honest measure of what a paged worker holds: torch "
            "cannot see aimdo memory, so without this the published number "
            "is wrong in the direction that causes host OOM",
     "since": None},

    # --------------------------------------------------------------- shared
    {"module": "comfy.model_management", "attr": "TOTAL_PINNED_MEMORY",
     "kind": "attr", "severity": DEGRADE, "tier": SHARED, "side": WORKER,
     "why": "pinned RAM census; absent, the host cannot see what workers pin",
     "since": "ComfyUI 5aa5ccc9, 2026-05-20"},
    {"module": "comfy.model_management", "attr": "free_pins",
     "kind": "callable", "severity": DEGRADE, "tier": SHARED, "side": WORKER,
     "why": "lets a worker return pinned RAM under host memory pressure",
     "since": "ComfyUI 5aa5ccc9, 2026-05-20"},
)


def _entry_applies(entry, side, tier_levels):
    if entry["side"] not in (side, BOTH) and side != BOTH:
        return False
    return entry["tier"] in tier_levels


def evaluate(present, side=BOTH, tiers=(FLOOR,)):
    """Turn a presence map into a verdict. Pure: no imports, no I/O.

    ``present`` maps "module.attr" to one of True (there and the right
    shape), False (missing or wrong shape) or None (unknown, e.g. the module
    could not be imported at all).

    Returns ``(ok, failures, notes)``. ``ok`` is False only when a FATAL
    entry is missing, because the house rule is to degrade on availability
    and fail loudly on correctness. Unknown is never a failure: a checker
    that cannot see the module must not manufacture a verdict.
    """
    failures = []
    notes = []
    for entry in CONTRACT:
        if not _entry_applies(entry, side, tuple(tiers)):
            continue
        key = "{}.{}".format(entry["module"], entry["attr"])
        state = present.get(key)
        if state is not False:
            continue
        line = "{} missing: {}".format(key, entry["why"])
        if entry["since"]:
            line += " (needs {})".format(entry["since"])
        if entry["severity"] == FATAL:
            failures.append(line)
        else:
            notes.append(line)
    return (not failures), failures, notes


def required_keys(side=BOTH, tiers=(FLOOR,)):
    """Every "module.attr" an evaluation of this side and these tiers reads."""
    return tuple(
        "{}.{}".format(e["module"], e["attr"])
        for e in CONTRACT if _entry_applies(e, side, tuple(tiers))
    )


def probe_present(keys, modules=None):
    """Resolve a presence map for ``keys`` against live modules.

    Imports happen inside this function so the module itself stays free of
    them. ``modules`` overrides the lookup for tests. Presence is checked BY
    NAME with ``hasattr``, never with ``getattr(..., default)``: a default is
    exactly how a renamed upstream symbol becomes a silently wrong number.
    """
    import importlib

    cache = {}
    present = {}
    for key in keys:
        module_name, _, attr = key.rpartition(".")
        if module_name not in cache:
            if modules is not None:
                cache[module_name] = modules.get(module_name)
            else:
                try:
                    cache[module_name] = importlib.import_module(module_name)
                except Exception:
                    cache[module_name] = None
        module = cache[module_name]
        if module is None:
            present[key] = None          # unknown, never a verdict
            continue
        if not hasattr(module, attr):
            present[key] = False
            continue
        wanted = next(
            (e for e in CONTRACT
             if e["module"] == module_name and e["attr"] == attr), None)
        if wanted is not None and wanted["kind"] == "callable":
            present[key] = callable(getattr(module, attr))
        else:
            present[key] = True
    return present


def check(side=BOTH, tiers=(FLOOR,), modules=None):
    """Evaluate the contract against the running host. Never raises."""
    keys = required_keys(side, tiers)
    try:
        present = probe_present(keys, modules=modules)
    except Exception as exc:                      # pragma: no cover
        return True, [], ["contract probe failed: {}".format(exc)]
    return evaluate(present, side=side, tiers=tiers)
