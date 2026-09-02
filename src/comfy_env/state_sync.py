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
from typing import Any, Callable, Dict, List, Optional, Tuple

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
            if seq <= getattr(p, "_residency_seq", -1):
                continue  # stale: an eviction reply already superseded this
            resident = int(entry.get("resident", 0))
            size = int(getattr(p, "size", 0)) or resident
            resident = max(0, min(resident, size))
            old = int(getattr(p.model, "model_loaded_weight_memory", 0))
            p._residency_seq = seq
            # admission ceiling: highest residency seen since the last direct
            # command receipt. _worker_held_bytes reads it so a paging model
            # cannot make true free look larger than it is.
            p._residency_peak = max(resident,
                                    int(getattr(p, "_residency_peak", 0)))
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
