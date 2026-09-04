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
| `p2_reserve_levers.py` | Which VRAM levers actually change host behaviour, on each memory path? | yes |
| `p3_prompt_epoch.py` | Can the prompt epoch be read instead of patched for? | no |

## What was measured

RTX 3090 (24576 MiB), ComfyUI 2026-08-24, comfy-aimdo 0.4.13.

- **P1**, 2026-09-04: with the protocol-level guard, a worker on 0.4.13
  against a parent reporting 0.4.15 keeps aimdo (both are protocol level
  3), while a genuine level difference is refused with
  `aimdo protocol skew: worker level 3 (0.4.13), parent level 4 (9.9.9)`.
  Before the fix the version strings were compared directly and two of
  nineteen environments on this machine were silently on the ledger.

- **P2**, 2026-09-04: the reserve is preventive on the legacy path and inert
  on the paged one. A 20.13 GiB reserve takes a 6 GiB model from 6.00 GiB
  resident to 1.38 GiB on the legacy path; on the paged path neither the
  ComfyUI reserve nor aimdo's own headroom setter moves residency at all
  (6.03 GiB in every case). aimdo's headroom is fixed at `init_devices`:
  the setter is inert once running, a second `init_devices` returns False,
  and a second `control.init` segfaults. The lever that does work there is
  reactive, `free_memory(target, device)`, which took the same model from
  6.03 GiB to 0.03 GiB.

  Any test of these levers must create genuine pressure. Three earlier
  versions of P2 reported a working lever as inert purely because the model
  still fitted, or because the eviction target was below free memory.

- **P3**, 2026-09-04: yes. `get_progress_state().prompt_id` reads None with
  no prompt running, carries ComfyUI's real prompt id once one starts, is
  stable within a prompt and changes across one. No hook installed. This
  replaced a class patch of `PromptModelTracker.start`.
