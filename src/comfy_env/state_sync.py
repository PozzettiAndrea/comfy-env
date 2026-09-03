"""Pure logic for the memory seam: residency census apply, node state filtering.

Shared by the parent (as ``comfy_env.state_sync``) and the worker (copied
beside the worker script and imported as ``state_sync``, the same staging as
``_ipc_shared`` and ``memory_manager``). Everything here is dict math with no
comfy or torch import, so bare CI drives it directly.

Design provenance: two independent 4 person groups converged on this shape and
cross examined each other's drafts; every rule below carries the reason it won.
"""

from __future__ import annotations

import hashlib
import pickle
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Problem 7: residency census
# ---------------------------------------------------------------------------

#: Parent side switch. "boundary" (default) applies censuses at node
#: boundaries; "command" ignores frame censuses and trusts only command echoes;
#: "off" restores registration-time pinning. Deliberately NO interval knob:
#: any timer would write while an upstream iteration may be in flight.
RESIDENCY_ENV_VAR = "COMFY_ENV_RESIDENCY_REFRESH"

#: Drift above this logs a WARN: a ledger that silently drifted by half a
#: gigabyte is the operator-visible symptom of every bug in this seam.
RESIDENCY_WARN_BYTES = 512 * 1024 * 1024


def apply_residency(patchers: Dict[str, Any], census: List[Dict[str, Any]],
                    log: Optional[Callable[[str], None]] = None) -> List[str]:
    """Apply a worker residency census to the parent's proxies. Returns the
    ids whose ledger value changed.

    Rules, each one a debate verdict:

    * REPLACE semantics: a census is a total, the last receipt wins. No delta
      thresholds on the wire (a threshold is a second place to be wrong).
    * Per-model ``seq``: an entry not newer than the last applied write for
      that model is dropped whole. This is the only defence against a frame
      census resurrecting bytes an eviction reply just freed.
    * Clamp to ``[0, size]``: the ledger never exceeds ``model_size()``.
    * Ids MISSING from a census keep their old value. Missing means unknown,
      not zero: zeroing would under-state residency, over-state true free via
      ``_worker_held_bytes``, and admit a load that OOMs.
    * The ledger value ROUNDS DOWN (it is a receipt); the pessimistic ceiling
      for admission lives in ``_worker_held_bytes`` via ``residency_peak``,
      comfy-env's own function, invisible to upstream.
    * Device flips at the 0 boundary, mirroring what a real detach does.
    * MUST be called outside any upstream iteration: ``free_memory`` snapshots
      its sort key (model_management.py:875) then re-reads inside the loop
      (:807); a write between those reads describes a model that no longer
      exists. The node boundary (``_register_new_patchers``) is the sanctioned
      site.
    """
    changed: List[str] = []
    for entry in census or []:
        mid = entry.get("id")
        p = patchers.get(mid)
        if p is None:
            continue
        try:
            seq = int(entry.get("seq", 0))
            resident = int(entry.get("resident", 0))
            size = int(getattr(p, "size", 0)) or resident
            resident = max(0, min(resident, size))
            old = int(getattr(p.model, "model_loaded_weight_memory", 0))
            if seq < getattr(p, "_residency_seq", -1):
                # Strictly older only: a census carrying the SAME seq as a
                # command echo was sampled AFTER that echo's byte movement
                # (the worker bumps at echo-send time, single threaded, one
                # FIFO socket), so the tie goes to the census. The old <=
                # drop made every post-echo census a dead letter: nothing
                # bumps seq at a lazy re-fault, so the eviction blindness
                # window never closed.
                if log is not None and \
                        abs(resident - held_ceiling(p)) > RESIDENCY_WARN_BYTES:
                    log(f"[comfy-env] WARNING stale census dropped model={mid} "
                        f"worker={resident / 1e9:.2f}GB "
                        f"ceiling={held_ceiling(p) / 1e9:.2f}GB "
                        f"(seq {seq} < {getattr(p, '_residency_seq', -1)}); "
                        f"parent may be blind to re-faulted bytes")
                continue
            p._residency_seq = seq
            # The census is the DECAY AUTHORITY for the admission peak: it is
            # a present-state statement sampled at a boundary, when the env is
            # about to be idle (no unsignaled regrowth exits; while a call IS
            # in flight the held_charge size override carries the pessimism,
            # not the peak). A peak that only ratcheted up strands the old
            # peak forever after real evictions.
            p._residency_peak = resident
            if resident == old:
                continue
            p.model.model_loaded_weight_memory = resident
            if resident <= 0:
                p.model.device = p.offload_device
            elif entry.get("device", "").startswith("cuda"):
                p.model.device = p.load_device
            changed.append(mid)
            if log is not None and abs(resident - old) > RESIDENCY_WARN_BYTES:
                log(f"[comfy-env] WARNING residency drift model={mid} "
                    f"ledger={old / 1e9:.2f}GB worker={resident / 1e9:.2f}GB "
                    f"seq={seq}")
        except Exception:
            continue  # one bad entry must not poison the census
    return changed


#: Fields of the unified per-frame VRAM report. Each is a TOTAL (replaces),
#: never a delta; a field absent from a frame means "unknown this frame".
VRAM_REPORT_FIELDS = ("residency", "overhead", "pinned")


def merge_vram_report(prev, frame):
    """Fold one worker frame's ``_vram_report`` into the stored one.

    Per field REPLACE; absent or None fields keep the stored value. Wrong
    implementations this pins: whole-dict replace (a frame that failed to
    sample overhead would erase the last good overhead, and the admission
    formula would silently drop the excess term); and treating None as a
    value (a fabricated zero). Returns a NEW dict; never mutates ``prev``.
    """
    merged = dict(prev or {})
    for key in VRAM_REPORT_FIELDS:
        val = (frame or {}).get(key)
        if val is not None:
            merged[key] = val
    return merged


def apply_echo(p: Any, resident: int, seq: Optional[int] = None,
               unmapped: bool = False, in_flight: bool = False) -> None:
    """Apply a command echo (partial load/unload reply, or a detach receipt).

    The ledger and seq are always written: an echo is the freshest receipt
    for what it proves. The PEAK obeys the admissibility criterion (lower
    the admission ceiling only when every regrowth exit out of the certified
    state is signaled):

    * ``unmapped`` (detach): peak drops to 0 in either flag state. Regrowth
      from unmapped passes through the budget RPC, a load echo, or a fresh
      registration; every exit signals. This also fixes the shipped bug
      where a detach zeroed the ledger but left the peak stuck high forever.
    * partial echo, NOT in flight: peak drops to the echoed resident. An
      idle worker cannot re-fault (faults are synchronous worker Python), so
      a commanded eviction of an idle env releases its over-reserve
      immediately.
    * partial echo, IN FLIGHT: peak only rises. The worker can lazily
      re-fault mid call with no signal; the pessimism for that window lives
      in ``held_charge``'s size override, but the pure layer must stay
      admissible standing alone, so the peak is not lowered here.
    """
    try:
        resident = max(0, int(resident))
        if seq is not None:
            p._residency_seq = int(seq)
        if unmapped:
            p._residency_peak = 0
            resident = 0
        elif not in_flight:
            p._residency_peak = resident
        else:
            p._residency_peak = max(resident,
                                    int(getattr(p, "_residency_peak", 0)))
        p.model.model_loaded_weight_memory = resident
        if resident <= 0:
            p.model.device = p.offload_device
    except Exception:
        pass  # an echo bookkeeping failure must not fail the eviction path


def apply_peak_raise(patchers: Dict[str, Any],
                     census: List[Dict[str, Any]]) -> None:
    """Peak-ONLY raise pass for call paths with no node boundary (the aiohttp
    route path): admission pessimism may rise at any time, but the ledger and
    seq stay boundary-only, so the census remains in place for the full apply
    at the env's next call_method. Never lowers, never writes seq or ledger,
    never consumes the census. Callers hold the worker's _mem_lock; a bare
    peak write outside this function is the lost-update the seam guard bans.
    Never raises."""
    for entry in census or []:
        try:
            p = patchers.get(entry.get("id"))
            if p is None:
                continue
            p._residency_peak = max(int(getattr(p, "_residency_peak", 0)),
                                    int(entry.get("resident", 0)))
        except Exception:
            continue


def held_charge(p: Any, in_flight: bool) -> int:
    """Admission charge for one proxy.

    Case split of the admissibility criterion: an IN FLIGHT worker charges
    the supremum (model size), so unsignaled lazy re-faults can never exceed
    the charge; an IDLE worker has no unsignaled regrowth exits (comfy_aimdo
    has no fault hook or background prefetch thread; faults are synchronous
    worker Python), so its receipts are authoritative and it charges
    ``held_ceiling``. Residual, accepted: a pack thread that keeps computing
    after its call returns (the ADR-0025 unhooked-allocation class)."""
    if in_flight:
        return max(held_ceiling(p), int(getattr(p, "size", 0)))
    return held_ceiling(p)


def held_ceiling(p: Any) -> int:
    """The admission-side view of one proxy's residency: the ledger value or
    the peak since the last command, whichever is higher. Stale-LOW here is the
    fatal direction (it over-states free and admits an OOM), so admission is
    pessimistic while the ledger stays a receipt."""
    return max(int(getattr(p.model, "model_loaded_weight_memory", 0)),
               int(getattr(p, "_residency_peak", 0)))


# ---------------------------------------------------------------------------
# Problem 11: node state return
# ---------------------------------------------------------------------------

#: sync (default) returns mutated state; off is the pre-2026-09 in-only wire.
STATE_ENV_VAR = "COMFY_ENV_NODE_STATE"

#: Per attribute cap. A self attribute above this is almost always a
#: rebuildable cache; it goes to the worker-held overflow tier, named, never
#: silently truncated and never shipped.
STATE_MAX_BYTES_ENV_VAR = "COMFY_ENV_NODE_STATE_MAX_BYTES"
STATE_MAX_BYTES_DEFAULT = 8 * 1024 * 1024

#: Parent-only bookkeeping keys. The seed sentinel is set by the PARENT on
#: first ingest and never crosses as state: a worker writing it would plant an
#: attribute the pack author never wrote.
SEED_SENTINEL = "_comfy_env_seeded"
STATE_ID_KEY = "_comfy_env_state_id"
RESERVED_KEYS = frozenset({SEED_SENTINEL, STATE_ID_KEY})

MARKER_KEY = "__comfy_env_overflow__"


def is_overflow_marker(value: Any) -> bool:
    return isinstance(value, dict) and value.get(MARKER_KEY) is True


def make_marker(gen: str, handle: int, name: str, reason: str,
                nbytes: int) -> Dict[str, Any]:
    """Parent-side stand-in for a value held in the worker. ``handle`` is a
    monotonic token, never id(): id() recycles after the sweep and a recycled
    key is silently someone else's state."""
    return {MARKER_KEY: True, "gen": gen, "handle": int(handle),
            "name": name, "reason": reason, "bytes": int(nbytes)}


def outbound_state(instance_dict: Dict[str, Any]) -> Dict[str, Any]:
    """What the parent sends: the instance dict minus parent-only keys."""
    return {k: v for k, v in instance_dict.items() if k not in RESERVED_KEYS}


def fingerprint(value: Any, cap: int) -> Tuple[Optional[str], str, int]:
    """(digest, verdict, nbytes) for one attribute value.

    verdict: "ship" (small, picklable), "overflow" (too big, unpicklable,
    device-resident, or a receipt), computed WORKER-side where both the pre-
    and post-call copies exist. Byte-level, not identity: identity-diff misses
    in-place mutation (``self.cache["x"] = 1`` mutates the same object), and
    ``!=`` on a tensor raises. That argument killed identity-diff in cross-exam.
    """
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None and int(nbytes) > cap:
        return None, "over_cap", int(nbytes)
    device = getattr(value, "device", None)
    if device is not None and getattr(device, "type", "") == "cuda":
        # shipping a CUDA tensor is a copy that breaks `is` checks and moves
        # VRAM-sized data per call; it stays worker-side.
        return None, "device_resident", int(nbytes or 0)
    if type(value).__name__ in ("OpaquePayload", "OpaquePickle"):
        # a receipt bounced back is a lie, not state
        return None, "worker_only_type", 0
    try:
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        return None, "unpicklable", 0
    if len(blob) > cap:
        return None, "over_cap", len(blob)
    return hashlib.sha256(blob).hexdigest(), "ship", len(blob)


def diff_state(pre: Dict[str, Any], post: Dict[str, Any], cap: int,
               gen: str, mint_handle: Callable[[], int],
               store: Callable[[int, Any], None]) -> Dict[str, Any]:
    """Worker-side: build ``self_state_out`` from the pre/post instance dicts.

    * ``set``: keys whose serialized bytes differ from the inbound copy, plus
      new keys. Unchanged keys are omitted (the parent keeps its copy), which
      is what bounds steady-state wire cost.
    * ``deleted``: inbound keys absent after the call, so a real ``del
      self.x`` propagates. Derived from sent keys, never from the diff, which
      keeps it lossless.
    * ``dropped``: every key that could not ship, with the reason and bytes.
      Overflow-eligible values are handed to ``store`` and represented in
      ``set`` by a marker so they round-trip on the next call.
    * State returns on error frames too: a non-isolated node that mutates
      ``self`` and then raises keeps the mutation.
    """
    pre_fp: Dict[str, Optional[str]] = {}
    for k, v in pre.items():
        if k in RESERVED_KEYS:
            continue
        if is_overflow_marker(v):
            pre_fp[k] = f"marker:{v.get('handle')}"
        else:
            pre_fp[k] = fingerprint(v, cap)[0]

    out_set: Dict[str, Any] = {}
    dropped: List[Dict[str, Any]] = []
    for k, v in post.items():
        if k in RESERVED_KEYS:
            continue
        if is_overflow_marker(v):
            # the worker resolved this inbound; if the attribute still holds
            # the marker object itself the pack never touched it
            continue
        digest, verdict, nbytes = fingerprint(v, cap)
        if verdict == "ship":
            if pre_fp.get(k) == digest:
                continue  # byte-identical: parent already holds it
            out_set[k] = v
        else:
            handle = mint_handle()
            store(handle, v)
            marker = make_marker(gen, handle, k, verdict, nbytes)
            out_set[k] = marker
            dropped.append({"name": k, "reason": verdict, "bytes": nbytes,
                            "marker": {"gen": gen, "handle": handle}})

    deleted = [k for k in pre
               if k not in RESERVED_KEYS and k not in post]
    return {"set": out_set, "deleted": deleted, "dropped": dropped}


def apply_state_out(instance_dict: Dict[str, Any],
                    state_out: Optional[Dict[str, Any]]) -> None:
    """Parent-side apply: setattr semantics for ``set``, delete for
    ``deleted``, never touch a key the worker did not mention. Sets the seed
    sentinel, because a state_out existing means the worker has run the real
    ``__init__`` (or its fallback) for this instance."""
    if not state_out:
        return
    for k, v in (state_out.get("set") or {}).items():
        if k not in RESERVED_KEYS:
            instance_dict[k] = v
    for k in state_out.get("deleted") or []:
        if k not in RESERVED_KEYS:
            instance_dict.pop(k, None)
    instance_dict[SEED_SENTINEL] = True


# ---------------------------------------------------------------------------
# Pinned RAM budgets (coverage sweep item 1)
# ---------------------------------------------------------------------------

#: Gate for the whole split. "off" (the shipped default) is byte-identical to
#: today; "auto" clamps worker pin ceilings from the host allowance. The
#: default flips only after the measurement gates re-run on real census data.
PIN_SPLIT_ENV_VAR = "COMFY_ENV_PIN_SPLIT"

#: Bootstrap grant for a worker that has not seen a budget reply yet (bytes).
PIN_SHARE_ENV_VAR = "COMFY_ENV_PIN_SHARE"

#: Mirror of the host's RAM_CACHE_HEADROOM (bytes). Workers otherwise sit on
#: the flat 2 GB floor at model_management.py:720 forever, because the only
#: setter (execution.py:748) never runs in a worker.
PIN_HEADROOM_ENV_VAR = "COMFY_ENV_PIN_HEADROOM"

PIN_FLOOR_ENV_VAR = "COMFY_ENV_PIN_FLOOR"
PIN_RESERVE_ENV_VAR = "COMFY_ENV_PIN_RESERVE"

#: Matches the ensure_pin_budget floor at model_management.py:720.
PIN_FLOOR_DEFAULT = 2 * 1024 ** 3

#: The budget owner's ADVANCE PAYMENT of the reserve margin: a worker parses
#: empty argv, so its EXTRA_RESERVED_VRAM sits on the upstream default from
#: spawn until the first budget reply, and the dtype selectors (which read it
#: at model creation, permanently) run in that window. The pool injects the
#: host's resolved value verbatim; the first reply's plain assignment
#: supersedes it. Same owner (the budget RPC), second channel for the
#: pre-RPC window only.
RESERVE_ENV_VAR = "COMFY_ENV_EXTRA_RESERVED_VRAM"
PIN_RESERVE_DEFAULT = 0.5

#: Damping (parent side): a grow below this delta is not emitted, so a paging
#: worker cannot retune the pool every node boundary.
PIN_DEADBAND_BYTES = 512 * 1024 * 1024

#: Consecutive censuses with an unchanged consumer set before a GROW is
#: emitted. Shrinks apply on the next reply, immediately: shrink-fast,
#: grow-slow is what bounds the sawtooth.
PIN_GROW_STABLE_CENSUSES = 2


def update_pin_reports(reports: Dict[str, Dict[str, int]], key: str,
                       pinned: int, seq: int) -> bool:
    """Ingest one pin report into the parent's ledger. Returns True if
    applied. Same stale-drop rule as ``apply_residency``: an entry not newer
    than the last applied write for that key is dropped whole, so an
    out-of-order frame cannot resurrect a stale total."""
    try:
        seq = int(seq)
        old = reports.get(key)
        if old is not None and seq <= int(old.get("seq", -1)):
            return False
        reports[key] = {"pinned": max(0, int(pinned)), "seq": seq}
        return True
    except Exception:
        return False


def allocate_pin_budgets(host_max: int, reports: Dict[str, Dict[str, int]],
                         floor_bytes: int = PIN_FLOOR_DEFAULT,
                         reserve: float = PIN_RESERVE_DEFAULT,
                         requester: Optional[str] = None) -> Dict[str, int]:
    """Split the host pin allowance across the processes that share its RAM.

    ``reports`` maps ``"host"`` plus worker keys to ``{"pinned": bytes,
    "seq": n}``. Dead workers' keys are ABSENT from the input (the pool
    removes them), so they are absent from the output; a missing census from
    a live worker keeps its last report, mirroring ``apply_residency``.

    Rules, each an invariant from the design debate:

    * ``host_max <= 0`` means pinning was never enabled anywhere: every key
      gets the ``-1`` sentinel unchanged, and nothing can be stranded.
    * The drain bound beats conservation: no grant ever lands below a
      holder's current pinned total (a lower ceiling makes the registration
      check at model_management.py:739 a permanent shortfall, evicting
      forever), and never 0 for anyone (unpin_memory early-returns on
      ``MAX <= 0``, stranding registrations). Grants may therefore exceed
      ``host_max`` by at most the sum of the floor and pinned overages;
      that is the documented exception, not a bug.
    * The share denominator counts LIVE pinners (pinned > 0) plus the
      requester, so idle workers land on the floor without draining the
      pool's working shares.
    """
    if host_max <= 0:
        return {k: -1 for k in reports}
    host_pinned = int(reports.get("host", {}).get("pinned", 0))
    grant_host = max(int(floor_bytes), host_pinned, int(host_max * reserve))
    workers = [k for k in reports if k != "host"]
    live = {w for w in workers
            if int(reports[w].get("pinned", 0)) > 0 or w == requester}
    share = (int(host_max) - grant_host) // max(1, len(live))
    out: Dict[str, int] = {}
    for k in reports:
        if k == "host":
            out[k] = grant_host
        elif k in live:
            out[k] = max(int(floor_bytes),
                         int(reports[k].get("pinned", 0)), share)
        else:
            # Idle and not asking: the floor (or what it still holds while
            # draining), never the share -- handing the live remainder to
            # every idle worker would multiply it, not split it.
            out[k] = max(int(floor_bytes), int(reports[k].get("pinned", 0)))
    return out


def damp_pin_grant(last_grant: Optional[int], new_grant: int,
                   stable_censuses: int) -> int:
    """Parent-side damping for one worker's grant.

    Shrink applies immediately (the drain bound in the allocator already
    keeps it above held bytes). Grow waits for ``PIN_GROW_STABLE_CENSUSES``
    consecutive censuses with an unchanged consumer set, and a grow smaller
    than the deadband is swallowed entirely, because each emitted delta
    retunes every worker's allocator behavior."""
    if last_grant is None:
        return new_grant
    if new_grant <= last_grant:
        return new_grant
    if new_grant - last_grant < PIN_DEADBAND_BYTES:
        return last_grant
    if stable_censuses < PIN_GROW_STABLE_CENSUSES:
        return last_grant
    return new_grant


# ---------------------------------------------------------------------------
# Worker VRAM overhead booking (coverage item: cast buffers vs the flat cost)
# ---------------------------------------------------------------------------

#: Per-worker fixed VRAM cost OUTSIDE the caching allocator: CUDA context plus
#: cuBLAS/cuDNN handles. torch.cuda.memory_reserved structurally cannot see
#: these, so the floor and the measured excess partition cleanly (floor =
#: outside the allocator, excess = inside it beyond registered residency) and
#: they ADD; max() would under-book by min(floor, excess). Measured 276 to
#: 300 MiB on Linux/RTX 3090 (2026-09); the old 250 MiB figure was a Windows
#: RTX 4060 Ti measurement.
WORKER_VRAM_FLOOR = 300 * 1024 * 1024

#: An overhead report above this WARNs at ingest (a 30 GB report on a 24 GB
#: card is junk arithmetic worker-side; the clamp direction is still overbook,
#: which can only starve admission, never OOM it).
OVERHEAD_WARN_BYTES = 4 * 1024 ** 3


#: Share of the device an overhead report may reach before the warning
#: fires on small cards (the absolute cap is OVERHEAD_WARN_BYTES).
OVERHEAD_WARN_SHARE = 0.15


def blind_free_is_process_local(platform: str) -> bool:
    """Whether ``torch.cuda.mem_get_info``'s free number is the calling
    process's budget (Windows/WDDM) or the whole device (everything else).

    Measured both ways: a WDDM sibling holding 13 GiB moved the parent's
    number 75 MiB (process-local, see ``_true_device_free``); a Linux parent
    with 12 GiB in three workers read 10.25 GiB, identical to NVML
    (device-wide, experiment B1). Subtracting the worker ledger from a
    device-wide number double-books every worker byte: live, that verdict
    called free_memory for 16 GiB on a card with 10 GiB free and evicted
    8 GiB of executing worker models to admit a 4 GiB load."""
    return str(platform or "").lower().startswith("win")


def overhead_warn_threshold(device_total: Optional[int]) -> int:
    """Warn threshold for one worker's measured overhead: the lesser of the
    absolute cap and a share of the device. Catches: a fixed 4 GiB threshold
    that never fires on an 8 GiB card where 2 GiB of allocator residue is
    already a quarter of everything. Unknown total keeps the absolute cap."""
    try:
        total = int(device_total or 0)
    except Exception:
        total = 0
    if total <= 0:
        return OVERHEAD_WARN_BYTES
    return min(OVERHEAD_WARN_BYTES, int(total * OVERHEAD_WARN_SHARE))


def pin_regression_line(name: str, state: Dict[str, Any],
                        last_seen: Dict[str, int]) -> Optional[str]:
    """Always-on pin-regression notice: one line each time a worker's
    ACTIVE-pin eviction counter grows, else None. Active evictions are the
    signal that the host's pin-pressure sweep stole pins from the model
    still executing (the thing prompt marks exist to prevent); plain
    evictions of idle models are normal churn and stay silent. Mutates
    ``last_seen[name]`` so a flat counter never re-logs, and a counter that
    went DOWN (epoch reset) re-arms instead of staying muted forever."""
    try:
        active = int(state.get("pins_evicted_active_bytes", 0) or 0)
    except Exception:
        return None
    prev = int(last_seen.get(name, 0))
    last_seen[name] = active
    if active <= prev:
        return None
    churn = state.get("pin_churn", 0)
    evicted = int(state.get("pins_evicted_bytes", 0) or 0)
    return (f"[comfy-env] PIN REGRESSION env={name} "
            f"active_evicted={active / 1e6:.0f}MB "
            f"(+{(active - prev) / 1e6:.0f}MB) total_evicted={evicted / 1e6:.0f}MB "
            f"churn={churn}")


def update_overhead_reports(reports: Dict[str, Dict[str, int]], key: str,
                            excess: int, seq: int,
                            log: Optional[Callable[[str], None]] = None,
                            warn_bytes: int = OVERHEAD_WARN_BYTES) -> bool:
    """Ingest one worker's measured overhead excess. Same stale-drop shape as
    ``update_pin_reports``: REPLACE on newer seq (the value is self-measured
    in-frame, so no peak is needed; stale-HIGH residue while idle is the safe
    over-book direction), drop older whole. ``warn_bytes`` is the caller's
    device-shaped threshold (see ``overhead_warn_threshold``)."""
    try:
        seq = int(seq)
        old = reports.get(key)
        if old is not None and seq <= int(old.get("seq", -1)):
            return False
        excess = max(0, int(excess))
        if log is not None and excess > int(warn_bytes):
            log(f"[comfy-env] WARNING worker overhead report {key}: "
                f"{excess / 1e9:.2f}GB exceeds {warn_bytes / 1e9:.2f}GB")
        reports[key] = {"excess": excess, "seq": seq}
        return True
    except Exception:
        return False


def held_from_snapshot(snapshot: Dict[str, Dict[str, Any]],
                       floor: int = WORKER_VRAM_FLOOR,
                       cap: Optional[int] = None) -> int:
    """Total worker-held VRAM from a pool snapshot. Pure; owned here so bare
    CI drives the whole admission arithmetic.

    ``snapshot`` maps worker key to ``{"in_flight": bool, "excess":
    Optional[int], "models": [{"ledger": int, "peak": int, "size": int}]}``.
    Dead workers' keys are ABSENT (the pool removes them); presence IS
    liveness, so every listed worker books the floor even with zero models
    (the old ``if patchers:`` skip booked a modelless live worker's CUDA
    context at zero).

    Per worker: sum over models of the per-patcher charge (``size`` when in
    flight, else ``max(ledger, peak)`` -- the in-flight charge REPLACES the
    ceiling, it never adds, or every resident byte of a computing worker
    would be counted twice), plus ``floor``, plus the measured excess clamped
    to ``[0, cap - floor]`` when a cap is known. The excess stays booked
    while in flight: suppressing it would under-book cast buffers and cache
    that physically persist through the call (the OOM direction); keeping it
    risks at most one stale excess overlapping the size charge, which
    over-books and heals on the completion frame.
    """
    total = 0
    for w in snapshot.values():
        in_flight = bool(w.get("in_flight"))
        for m in w.get("models") or []:
            if in_flight:
                total += max(int(m.get("size", 0)),
                             int(m.get("ledger", 0)), int(m.get("peak", 0)))
            else:
                total += max(int(m.get("ledger", 0)), int(m.get("peak", 0)))
        excess = w.get("excess")
        excess = max(0, int(excess)) if excess is not None else 0
        if cap is not None:
            excess = min(excess, max(0, int(cap) - int(floor)))
        total += int(floor) + excess
    return total


def forward_cast_need(largest_tensor: Optional[int],
                      num_streams: Optional[int]) -> int:
    """The future cast-buffer bytes of the load being admitted RIGHT NOW.

    Cast buffers allocate lazily at the first forward, AFTER admission, so
    neither NVML nor the measured excess (both report the past or present)
    can see them; only the incoming load's own numbers can. ``largest_tensor``
    comes from the worker's shim (it holds the real patchers), and
    ``num_streams`` is the worker's LIVE resolved value sent on the same
    request (never cli_mirror). Degrades: either input absent means 0, which
    collapses the need formula's max() back to min_inference, today's
    behavior. The consumer takes max(min_inference, this), never the sum:
    cast buffers are inference transients competing for the same reserve.
    """
    if not largest_tensor or not num_streams:
        return 0
    return max(0, int(num_streams)) * max(0, int(largest_tensor))


# ---------------------------------------------------------------------------
# /free broadcast planning
# ---------------------------------------------------------------------------

#: Minimum seconds between release broadcasts. Suppresses SAME-BURST
#: duplicates only (a nested double-wrap, an OOM retry storm): deliberately
#: short, because the wrap site cannot distinguish a human /free from OOM
#: recovery (both call unload_all_models), so a long window would swallow a
#: genuine press arriving right after an OOM broadcast, leaving those
#: workers with only the shallow detach and not the deep ladder. Accepted
#: residual: a human press within this window still gets the shallow pass.
RELEASE_DEBOUNCE_SECONDS = 0.5


def plan_release_broadcast(workers: Dict[str, Dict[str, Any]], now: float,
                           last_broadcast: float,
                           debounce: float = RELEASE_DEBOUNCE_SECONDS
                           ) -> Dict[str, List[str]]:
    """Which workers get the full_release command.

    ``workers`` maps key to ``{"alive": bool, "advertises": bool}``. Every
    input key lands in exactly one output list (set equality, so a filtered
    worker cannot hide): ``send`` (alive advertisers), ``skip_dead`` (their
    memory died with them; the send path must NEVER resurrect one to free
    it), ``skip_unsupported`` (no ready-frame advertisement; an unknown
    method gets no reply and the sender eats the recv timeout). Inside the
    debounce window everything moves to ``skip_debounced``.
    """
    out: Dict[str, List[str]] = {"send": [], "skip_dead": [],
                                 "skip_unsupported": [], "skip_debounced": []}
    if now - last_broadcast < debounce:
        out["skip_debounced"] = sorted(workers)
        return out
    for key, w in workers.items():
        if not w.get("alive"):
            out["skip_dead"].append(key)
        elif not w.get("advertises"):
            out["skip_unsupported"].append(key)
        else:
            out["send"].append(key)
    return out


# ---------------------------------------------------------------------------
# Prompt-epoch pin marks (worker current_prompt protection)
# ---------------------------------------------------------------------------

#: One kill switch, both ends: gates the host-side PromptModelTracker patch
#: AND all worker mark writes. Off restores byte-identical behavior.
PIN_MARKS_ENV_VAR = "COMFY_ENV_PIN_MARKS"

#: Sticky-fallback decay: when no epoch token arrives (the host patch failed
#: on an upstream refactor), marks survive this many calls and then clear.
#: A named constant, deliberately not a knob: dark degrade would silently
#: reopen the corruption window (tier 1 ignores `active`, and an unmarked
#: mid-load model's staging pages can be decommitted under an in-flight
#: async copy) exactly when protection matters most; sticky-with-decay keeps
#: the window closed at the cost of bounded stale-mark overhang, visible in
#: pin_churn.
PROMPT_MARK_DECAY_CALLS = 8

def clear_on_epoch_change(marks: Dict[str, Any], gen: Optional[int],
                          call_n: int, live: Iterable[str],
                          decay_calls: int = PROMPT_MARK_DECAY_CALLS
                          ) -> Tuple[Dict[str, Any], List[str]]:
    """Which marks to retire at the top of a worker call.

    ``marks`` maps model key to ``(gen, call_n)`` of its last marking.
    Returns ``(new_marks, to_clear)``; ``to_clear`` are keys whose
    ``current_prompt`` flag must flip False. Rules:

    * A mark from a DIFFERENT epoch clears: the previous prompt ended, and
      protecting its models would demote this prompt's models behind them in
      the tier walk (priority inversion).
    * ``gen None`` is the sticky fallback: marks survive ``decay_calls``
      calls from their stamping and then clear. Never indefinite (marks
      without clears are forbidden) and never instant (dark would reopen
      the corruption window on a host-patch failure).
    * Keys absent from ``live`` are pruned WITHOUT a flip: the object is
      gone; emitting a clear for it would make the applier chase ghosts.
    """
    live = set(live)
    new_marks: Dict[str, Any] = {}
    to_clear: List[str] = []
    for key, (mgen, mcall) in marks.items():
        if key not in live:
            continue  # pruned silently
        if gen is not None:
            if mgen == gen:
                new_marks[key] = (mgen, mcall)
            else:
                to_clear.append(key)
        else:
            if call_n - mcall < decay_calls:
                new_marks[key] = (mgen, mcall)
            else:
                to_clear.append(key)
    return new_marks, to_clear


def mark_on_load(marks: Dict[str, Any], gen: Optional[int], call_n: int,
                 loading_ids: Iterable[str]
                 ) -> Tuple[Dict[str, Any], List[str]]:
    """Mark the models a load is about to touch, BEFORE the load runs.

    The pressure fires during the load itself (``ensure_pin_registerable``
    inside ``pin_memory``), and pin tier 1 is ``(cp=False, active=None)``,
    ``active`` NOT consulted, so marking at call end would leave every
    model's first loaded call exposed to the decommit-under-async-copy
    window. Returns ``(new_marks, to_set)``: ``to_set`` are keys whose flag
    must flip True (keys already marked keep their True flag and migrate
    epochs WITHOUT a flip; a clear/set pair there would open a one-tier gap
    mid-walk). Every loading id gets a fresh ``(gen, call_n)`` stamp either
    way, which is what feeds the sticky fallback's decay clock."""
    new_marks = dict(marks)
    to_set: List[str] = []
    for key in loading_ids:
        if key not in new_marks:
            to_set.append(key)
        new_marks[key] = (gen, call_n)
    return new_marks, to_set


# ---------------------------------------------------------------------------
# Host RAM-pressure pin reclaim planning
# ---------------------------------------------------------------------------

#: Workers pinning less than this are skipped by a pressure sweep: the repin
#: cost (~200 ms per GiB measured) is not worth sub-slice reclaims.
PIN_PRESSURE_MIN_SLICE = 256 * 1024 * 1024

#: Minimum seconds between pressure sweeps. The host's own sweep runs once
#: per completed node; without a limit a sustained-pressure prompt would
#: shred and repin worker buffers every node.
PIN_PRESSURE_INTERVAL = 10.0


def plan_pin_pressure(reports: Dict[str, Dict[str, int]], target: int,
                      now: float, last_sweep: float,
                      min_slice: int = PIN_PRESSURE_MIN_SLICE,
                      interval: float = PIN_PRESSURE_INTERVAL
                      ) -> Dict[str, int]:
    """How many pinned bytes to ask each worker to release under host RAM
    pressure. Proportional to each worker's pinned census (never all of it
    from one victim), skipping workers under ``min_slice``; empty inside the
    rate-limit window or when nothing is reclaimable. The "host" key is the
    host's own report and is never asked (upstream's own free_pins handles
    the host side on the same trigger)."""
    if now - last_sweep < interval or target <= 0:
        return {}
    candidates = {k: int(r.get("pinned", 0)) for k, r in reports.items()
                  if k != "host" and int(r.get("pinned", 0)) >= min_slice}
    total = sum(candidates.values())
    if total <= 0:
        return {}
    ask_total = min(int(target), total)
    return {k: max(min_slice, ask_total * pinned // total)
            for k, pinned in candidates.items()}
