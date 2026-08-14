# CLAUDE.md -- comfy-env

## Project Overview

**comfy-env** is a dependency isolation system for ComfyUI custom node
packs. Each pack's nodes run in a persistent subprocess with its own
pixi-managed environment; the parent talks to synthesized proxy classes
over socket IPC with shared memory for bulk data. Architecture docs and
ADRs live in the separate `comfy-forge-docs` repo (published at
pozzettiandrea.github.io/comfy-forge-docs/) -- **the ADRs are the
authoritative "why"; this file is just orientation.**

## Layout (isolation side)

- `src/comfy_env/isolation/wrap.py` -- `register_nodes()`: config
  discovery, metadata scan, proxy synthesis, worker pool (one worker per
  env, generation counter on restart), VRAM budget negotiation,
  `[types]` validation + serializer loading (ADR-0015).
- `src/comfy_env/isolation/workers/subprocess.py` -- parent-side
  `SubprocessWorker`: spawn, health check, `call_module`/`call_method`
  (600 s default timeout -- see ADR-0018), consumed-ack after each read.
- `src/comfy_env/isolation/workers/_ipc_parent.py` -- parent transport
  internals: `SocketTransport`, tensor strategies, `_from_shm`.
- `src/comfy_env/isolation/workers/_persistent_worker.py` -- the worker
  program. Never imported by the parent: read as text and run by the
  isolated interpreter (ADR-0006), with `_ipc_shared.py` copied
  alongside. Must stay parseable by the OLDEST worker-env Python (3.9).
- `src/comfy_env/isolation/workers/_ipc_shared.py` -- the shared
  serialization core both sides import: `_to_shm_generic` walker,
  serializer registry (`register_serializer`), `OpaquePayload`/
  `OpaquePickle` (materialize-on-receipt, 0.4.15), pickle-frame helpers,
  `load_serializer_files`.

Install side: `config/` (toml parsing incl. root `[types]`),
`packages/` (pixi bootstrap -- pinned + sha256, `toml_generator.py`
manifest compiler, CUDA wheels, node deps), `install/` (workspace
materialization), `environment/`, `detection/`.

## Serialization ladder (ADR-0005)

CudaIPC (Linux GPU) > PoolIPC (cudaMallocAsync-safe GPU, default-off) >
TensorRef (torch shm, CPU) > numpy-via-torch > pickle block (loud
TypeError if pickling fails -- never leaks raw objects) > JSON
primitives. Pack types bypass pickle via the registry: `[types]` in
`comfy-env-root.toml` + `serialization.py` (ADR-0014/0015). A side that
cannot reconstruct a type holds a MATERIALIZED receipt (owned bytes,
survives worker restarts). A canary echo verifies the production
transport per worker at startup; probes fail closed.

## Lifetime protocol (0.4.15)

Worker keeps a call's shm blocks/tensors until the parent sends
`{"type": "consumed", "call_id": N}` after reading the reply;
`TENSOR_KEEPER_TTL` (60 s) survives only as the crash fallback. Do not
reintroduce timer-based correctness.

## Development

```powershell
cd D:\utils\comfy-env          # this machine; repo is cross-platform
./.venv-test/Scripts/python.exe -m pytest tests/ -q
```

Tests spawn REAL workers with `sys.executable` (no pixi needed) --
`tests/test_worker_roundtrip.py`, `test_serializer_registry.py`,
`test_transport_lifetime.py` are the transport contract. CI: 3-OS x
2-Python matrix; publish to PyPI is gated on green (pushes go to main).

## House rules

- Pre-1.0: no backward compatibility (ADR-0017) -- breaking changes
  remove the old way in the same release; comfy-env + affected packs
  ship together ("barrage"). Ends at the slow-rollout tripwire.
- Host-env principle: the ComfyUI env installs comfy-env and NOTHING
  else. The parent must never need a pack's libraries.
- Degrade on availability (missing env/GPU -> slower correct path),
  fail loudly on correctness (bad payloads, corrupt transport) --
  ADR-0008 as amended.
- No Co-Authored-By trailers in commits.
