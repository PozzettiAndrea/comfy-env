# Worker Architecture Overview

Isolated nodes run in persistent subprocess workers — separate Python processes with their own interpreter and packages. The parent ComfyUI process communicates with workers over Unix domain sockets (TCP on Windows) using length-prefixed JSON messages.

## Key Files

| File | Role |
|------|------|
| `isolation/wrap.py` | Worker pool, node registration, callback dispatch |
| `isolation/workers/subprocess.py` | SubprocessWorker, IPC protocol, serialization |
| `isolation/workers/base.py` | Worker ABC, WorkerError |
| `isolation/tensor_utils.py` | Tensor lifecycle helpers |
| `isolation/model_patcher.py` | SubprocessModelPatcher for VRAM coordination |
| `isolation/metadata.py` | Metadata extraction via subprocess |

## Architecture

```
ComfyUI (parent process)
│
├── Node A (main process)     ← imported normally
│
├── Node B (isolated)         ← proxy class
│   └── SubprocessWorker ◄──────────► Worker Process B
│       Unix socket / TCP              (own Python, own packages)
│
└── Node C (isolated)         ← proxy class
    └── SubprocessWorker ◄──────────► Worker Process C
        Unix socket / TCP              (own Python, own packages)
```

One persistent `SubprocessWorker` per isolation environment, reused across all calls. Workers auto-restart on crash.

## IPC Protocol

Messages are **length-prefixed JSON** over the socket:

```
[4-byte big-endian length][JSON payload]
```

Max message size: 100MB. Socket access is thread-safe (send/recv locks).

Large data (tensors, arrays) are **not** sent through the socket — they're transferred via shared memory, with only metadata pointers in the JSON messages. See [Serialization](worker-serialization.md).

## Message Flow

A typical `call_method` request:

```
Parent                              Worker
  │                                   │
  │─── serialize inputs to shm ──►    │
  │─── {type: call_method, ...} ──►   │
  │                                   │── deserialize inputs from shm
  │                                   │── execute node function
  │                                   │
  │    ◄── {type: callback,           │── (optional) request VRAM budget
  │         method: request_vram}     │
  │─── {type: callback_response} ──►  │
  │                                   │
  │    ◄── {type: callback,           │── (optional) report progress
  │         method: report_progress}  │
  │─── {type: callback_response} ──►  │
  │                                   │
  │                                   │── serialize outputs to shm
  │    ◄── {status: ok, result: ...}  │
  │── deserialize outputs from shm    │
  │── cleanup shared memory           │
```

The parent loops receiving messages until it gets a response with `status`. Intermediate messages are either **log forwarding** (`{type: log}`) or **callbacks** that get dispatched and answered inline.

## What Each Doc Covers

- [Serialization](worker-serialization.md) — how tensors, arrays, and objects cross process boundaries via shared memory
- [Memory Management](worker-memory.md) — VRAM budget negotiation, model tracking, device moves
- [Lifecycle](worker-lifecycle.md) — startup, health checks, crash recovery, progress reporting, shutdown
