"""The host-flag mirror: which CLI args cross to workers, and how.

A worker parses an EMPTY argv, so every ``comfy.cli_args`` flag resolves to
its default there: a host running fp8 gets workers loading the same weights
in fp16 (2x the footprint), a ``--disable-smart-memory`` host gets workers
that cache across prompts, a ``--disable-async-offload`` host gets the cast
buffers back. The parent serializes RESOLVED VALUES (never argv: an argv
replay re-parses against a possibly different cli_args.py, dies with
SystemExit(2) on version skew, and leaks through /proc/<pid>/cmdline) into
one env var; the worker applies them by setattr before its first comfy
import freezes the module-level reads.

Shared by the parent (as ``comfy_env.mirrored_args``) and the worker (staged
beside the worker script, imported as ``mirrored_args`` -- the same pattern
as ``state_sync`` and ``memory_manager``). Pure: no comfy import at module
level, 3.9-parseable, bare CI drives it directly.

The ALLOWLIST IS THE CONTRACT. Never mirrored, each for a stated reason:
``lowvram/novram/highvram`` (vram_state already crosses per call on the
budget RPC; a second authority can disagree with it), ``cuda_device`` (the
worker inherits the host's already-narrowed CUDA_VISIBLE_DEVICES; the flag
would re-index a second time), ``cpu`` (COMFY_ENV_COMFY_CPU/COMFY_CPU owns
it), ``reserve_vram`` (already crosses twice; the budget RPC is its owner),
``cache_*`` (executor-side; workers run no prompt queue), and every listen,
port, auth, and path flag (host server surface).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Optional

#: The serialized payload. JSON of {dest: resolved_value}.
MIRROR_ENV_VAR = "COMFY_ENV_HOST_ARGS"

#: Global kill switch: "0" disables the whole mirror (pack [env_vars] cannot
#: unset an args write, so this must exist).
MIRROR_KILL_ENV_VAR = "COMFY_ENV_MIRROR_ARGS"

#: Per-flag escape hatch: comma list of dests to withhold. Exists because the
#: global switch is too big a hammer: escaping one fast_disk regression with
#: it would also surrender the fp8 dtype mirror and reinstate the 2x
#: footprint (the inter-group blast-radius ruling).
NO_MIRROR_ENV_VAR = "COMFY_ENV_NO_MIRROR"

#: "auto" restores the worker's old attention auto-probe (for pack envs
#: richer than the host). Default: the worker follows the host.
WORKER_ATTENTION_ENV_VAR = "COMFY_ENV_WORKER_ATTENTION"

#: dest names on comfy.cli_args.args, hasattr-validated on both sides so a
#: ComfyUI without a flag degrades silently instead of erroring.
MIRRORED_ARGS = (
    # dtype and numerics (the 2x-footprint class, plus reproducibility)
    "force_fp32", "force_fp16",
    "fp32_unet", "fp64_unet", "bf16_unet", "fp16_unet",
    "fp8_e4m3fn_unet", "fp8_e5m2_unet", "fp8_e8m0fnu_unet",
    "fp16_vae", "fp32_vae", "bf16_vae", "cpu_vae",
    "fp8_e4m3fn_text_enc", "fp8_e5m2_text_enc",
    "fp16_text_enc", "fp32_text_enc", "bf16_text_enc",
    "fp16_intermediates", "supports_fp8_compute",
    "deterministic", "fast",
    "force_channels_last", "force_non_blocking",
    # memory behavior frozen at comfy.model_management import
    "disable_smart_memory", "disable_pinned_memory",
    "async_offload", "disable_async_offload",
    "fast_disk", "high_ram",
    "disable_mmap", "mmap_torch_files",
    # device-placement semantics the vram_state RPC does not carry
    # (text-encoder/VAE device choice); the other vram flags never mirror.
    "gpu_only",
)

#: Payload key for the host's resolved attention backend. Not an args dest:
#: the worker applies it at its attention site with an importability check.
ATTENTION_KEY = "attention"

#: Host flags that are deliberately NOT mirrored but change memory behavior;
#: reported so a --novram host sees the gap instead of silently eating it.
_VISIBILITY_FLAGS = ("lowvram", "novram", "highvram")


def _coerce(value: Any) -> Any:
    """JSON-safe rendering of a resolved arg value. Enums become their
    .value; sets become sorted lists (deterministic for the hash)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (set, frozenset)):
        return sorted(_coerce(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    inner = getattr(value, "value", None)
    if inner is not None and isinstance(inner, (bool, int, float, str)):
        return inner
    return str(value)


def parse_denylist(raw: Optional[str]) -> set:
    """COMFY_ENV_NO_MIRROR="fast_disk,high_ram" -> {"fast_disk","high_ram"}."""
    if not raw:
        return set()
    return {p.strip() for p in raw.split(",") if p.strip()}


def resolve_host_attention(args: Any) -> Optional[str]:
    """The host's RESOLVED backend, not a store_true flag: host-False is
    indistinguishable from host-default on the wire, and a host that could
    import sage but resolved pytorch attention made a deliberate choice the
    worker must not upgrade past."""
    if getattr(args, "use_sage_attention", False):
        return "sage"
    if getattr(args, "use_flash_attention", False):
        return "flash"
    return None


def serialize_host_args(args: Any, deny: Iterable[str] = ()) -> Dict[str, Any]:
    """Parent side: the payload for MIRROR_ENV_VAR. Allowlisted dests only,
    values coerced, denied names withheld. Never argv, never vars(args)."""
    deny = set(deny)
    payload: Dict[str, Any] = {}
    for name in MIRRORED_ARGS:
        if name in deny or not hasattr(args, name):
            continue
        payload[name] = _coerce(getattr(args, name))
    att = resolve_host_attention(args)
    if att is not None and ATTENTION_KEY not in deny:
        payload[ATTENTION_KEY] = att
    return payload


def apply_host_args(args: Any, payload: Dict[str, Any],
                    log=None) -> Dict[str, Any]:
    """Worker side: setattr each allowlisted payload entry onto the worker's
    args object. Returns {"applied": [names], "skipped": [{name, reason}]}.

    * Unknown or non-allowlisted keys are SKIPPED with a reason, never
      raised and never applied: version skew in either direction degrades,
      and the allowlist is enforced here too, so a hand-built payload cannot
      smuggle a non-contract flag through the worker.
    * ``fast`` re-hydrates to comfy's PerformanceFeature enum set; a member
      this comfy does not know skips the whole flag (half a set would be an
      invented value).
    * ``async_offload`` None stays None: comfy branches on ``is not None``
      (model_management.py:1341), so a None collapsed to 0 or "" would
      silently flip the stream count.
    * ATTENTION_KEY is not an args dest; the worker's attention site owns it.
    """
    _log = log or (lambda *_: None)
    applied = []
    skipped = []
    for name, value in sorted((payload or {}).items()):
        if name == ATTENTION_KEY:
            continue
        if name not in MIRRORED_ARGS:
            skipped.append({"name": name, "reason": "not_in_allowlist"})
            continue
        if not hasattr(args, name):
            skipped.append({"name": name, "reason": "unknown_here"})
            continue
        if name == "fast" and isinstance(value, list):
            try:
                from comfy.cli_args import PerformanceFeature
                value = set(PerformanceFeature(v) for v in value)
            except Exception as exc:
                skipped.append({"name": name,
                                "reason": f"unhydratable:{exc}"})
                continue
        try:
            setattr(args, name, value)
            applied.append(name)
        except Exception as exc:
            skipped.append({"name": name, "reason": f"setattr:{exc}"})
    if applied:
        _log(f"[worker] mirrored host args: {', '.join(applied)}")
    for s in skipped:
        _log(f"[worker] mirror skipped {s['name']}: {s['reason']}")
    return {"applied": applied, "skipped": skipped}


def readback_hash(args: Any, names: Iterable[str]) -> str:
    """Divergence hash: sha256 over the values READ BACK off the args object
    after apply, never over the received payload (a payload echo always
    matches its source and detects nothing). Both sides run this same
    function over the same names; equal hashes mean equal resolved values."""
    pairs = [[n, _coerce(getattr(args, n, None))] for n in sorted(set(names))]
    blob = json.dumps(pairs, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def unmirrored_nondefault(args: Any) -> list:
    """Deliberately unmirrored flags the host has set, for the ready-frame
    report: a --novram host sees the gap named instead of eating it."""
    return [n for n in _VISIBILITY_FLAGS if getattr(args, n, False)]
