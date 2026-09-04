"""One switch for worker memory management: COMFY_ENV_MEMORY_MANAGEMENT.

An ordered level rather than a set of independent flags, because the
features are not independent. Every pin feature is downstream of paging:
ComfyUI's eviction walk skips models that are not dynamic, so on a
non-paging worker there is nothing to mark and nothing to reclaim. A set of
switches could express combinations that cannot exist, and did: the shipped
code had seven separate variables and no way to say "whatever this host can
actually support".

Levels, lowest first:

  off      comfy-env manages nothing. The worker loads models the way any
           process would. Zero pinned RAM, which is a real advantage on a
           RAM-poor machine, and the widest compatibility of any level.
  ledger   ComfyUI's own low-VRAM streaming, plus the reserve and idle
           release. Big models still run, just slower than paged.
  paged    comfy-aimdo pages weights per layer, so a worker holds a fraction
           of the model resident. Prompt marks come WITH this level rather
           than above it: upstream reaches its pin eviction ladder on every
           host-buffer pin and its first tier matches every model not marked
           as the current prompt's, so paging without marks lets a worker's
           own running model be the first victim.
  shared   the worker also returns pinned host RAM when the machine is short
           of it.

``auto`` resolves to the highest level the host can actually support and
says so when it has to drop. An explicitly requested level that the host
cannot support still runs at what it can, loudly: refusing to start a pack
because one memory feature is unavailable would be failing on availability,
which the house rules forbid.

Pure module: no imports, no I/O. Exercised entirely in bare CI.
"""

OFF = "off"
LEDGER = "ledger"
PAGED = "paged"
SHARED = "shared"
AUTO = "auto"

#: Lowest to highest. Membership order IS the comparison.
ORDER = (OFF, LEDGER, PAGED, SHARED)

ENV_VAR = "COMFY_ENV_MEMORY_MANAGEMENT"

#: What each level needs from the host, as fact keys the caller supplies.
REQUIREMENTS = {
    OFF: (),
    LEDGER: (),
    PAGED: ("aimdo_available", "marks_available"),
    SHARED: ("aimdo_available", "marks_available", "pressure_available"),
}

#: Why a level is unreachable, in the operator's terms.
_REASONS = {
    "aimdo_available": "comfy-aimdo is not usable in this worker environment",
    "marks_available": (
        "this ComfyUI cannot mark the running prompt's models, so paging "
        "would let a worker's own model be the first eviction victim "
        "(needs ComfyUI c0117553, 2026-07-28)"
    ),
    "pressure_available": (
        "this ComfyUI has no RAM pressure hook to observe "
        "(needs ComfyUI b7a648ca, 2026-07-09)"
    ),
}


def rank(level):
    """Position in ORDER, or -1 for anything unknown."""
    try:
        return ORDER.index(level)
    except ValueError:
        return -1


def highest_supported(facts):
    """The best level this host can actually run."""
    best = OFF
    for level in ORDER:
        if all((facts or {}).get(key) for key in REQUIREMENTS[level]):
            best = level
    return best


def resolve(requested, facts):
    """Decide the level to run. Returns ``(level, note)``.

    ``note`` is None when nothing needs saying. It is populated when the
    resolved level is below what was asked for, which is the case an
    operator must be able to see: an unrequested demotion is the state that
    left environments silently running a different memory manager than
    their host.

    A requested level that is reachable resolves silently, including a
    deliberate downgrade. Choosing `ledger` on a RAM-poor box is a decision,
    not a fault, and warning about it would train the operator to ignore the
    channel that carries the real demotions.
    """
    requested = (requested or AUTO).strip().lower() or AUTO
    supported = highest_supported(facts)

    if requested == AUTO:
        note = None
        if supported != ORDER[-1]:
            note = "auto selected {}: {}".format(
                supported, _first_missing_reason(supported, facts))
        return supported, note

    if rank(requested) < 0:
        return supported, (
            "unknown level {!r}; running {} instead. Valid: {}".format(
                requested, supported, ", ".join(ORDER + (AUTO,))))

    if rank(requested) <= rank(supported):
        return requested, None

    return supported, (
        "{} was requested but this host supports {}: {}".format(
            requested, supported, _first_missing_reason(requested, facts)))


def _first_missing_reason(level, facts):
    """Why the level above ``level`` is not available."""
    nxt = rank(level) + 1
    if nxt >= len(ORDER):
        return "everything requested is available"
    for key in REQUIREMENTS[ORDER[nxt]]:
        if not (facts or {}).get(key):
            return _REASONS.get(key, key + " is unavailable")
    return "requirements for {} are met".format(ORDER[nxt])


def derived_flags(level):
    """The per-feature switches a level implies.

    These are what the worker and the host actually read. Keeping the
    mapping here means the level is the single source of truth and the
    individual switches cannot drift apart from it, which is how a half-off
    state (marks without paging) became possible before.
    """
    return {
        "aimdo": rank(level) >= rank(PAGED),
        "marks": rank(level) >= rank(PAGED),
        "pressure": rank(level) >= rank(SHARED),
        "reserve": rank(level) >= rank(LEDGER),
        "idle_release": rank(level) >= rank(LEDGER),
    }
