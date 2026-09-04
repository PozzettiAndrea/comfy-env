# Memory floor experiments

Standalone scripts behind comfy-env's memory redesign. Nothing here is
imported by `comfy_env`; each script answers a question the design asserts
an answer to, against a real ComfyUI tree and a real pack environment.

This directory exists because the equivalent harness was twice lost to
temp-directory cleanup. Experiments live in the repo now.

## Running

Every script takes its configuration from the environment, with defaults
for this machine:

| variable | meaning |
|---|---|
| `COMFY_DIR` | a ComfyUI source tree |
| `WORKER_PY` | the interpreter a worker runs |
| `COMFY_ENV_SRC` | which `comfy_env` to import |

Run with the pack environment's python, from any directory:

```
<pack env python> research/memory-floor/p1_aimdo_skew.py
```

Units are bytes everywhere they cross a function boundary; format only at
the edge. A previous round mixed GiB and 1e9 and its figures could not be
compared.

| script | question it answers | needs a GPU |
|---|---|---|
| `p1_aimdo_skew.py` | Does a comfy-aimdo patch bump on the host strand a worker on the legacy ledger? | yes |

## What was measured

RTX 3090 (24576 MiB), ComfyUI 2026-08-24, comfy-aimdo 0.4.13.

- **P1**, 2026-09-04: with the protocol-level guard, a worker on 0.4.13
  against a parent reporting 0.4.15 keeps aimdo (both are protocol level
  3), while a genuine level difference is refused with
  `aimdo protocol skew: worker level 3 (0.4.13), parent level 4 (9.9.9)`.
  Before the fix the version strings were compared directly and two of
  nineteen environments on this machine were silently on the ledger.
