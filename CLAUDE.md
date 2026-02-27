# CLAUDE.md — comfy-env

## Project Overview

**comfy-env** is a dependency isolation system for ComfyUI custom nodes. It runs each custom node's code in a separate subprocess with its own virtual environment (venv/pixi), communicating via socket-based IPC with shared memory for zero-copy tensor transfer.

## Key Architecture

### Process Model
- **Parent process**: ComfyUI main process. Runs `comfy.model_management`, dispatches node execution to workers.
- **Worker subprocess**: Isolated Python process with its own venv. Runs custom node code. Connected to parent via Unix domain socket (AF_UNIX) with length-prefixed JSON messages.
- **Tensor transfer**: Large data (tensors, arrays, meshes) goes through shared memory (/dev/shm). JSON messages carry metadata only.

### Critical File: `src/comfy_env/isolation/workers/subprocess.py`
This is the core of the IPC system (~2500 lines). It contains:
- **Parent-side code** (module level): `SocketTransport`, `_to_shm()`, `_from_shm()`, `_serialize_cuda_ipc()`, `_deserialize_cuda_ipc()`, `_probe_cuda_ipc()`, `SubprocessWorker` class
- **Worker-side code** (embedded as `_PERSISTENT_WORKER_SCRIPT` string): Duplicates of the serialization functions, worker main loop, ComfyUI integration patches

Both sides have their own copies of serialization logic because the worker runs in a separate Python process with potentially different packages.

### Tensor Serialization Strategies (priority order)
1. **CUDA IPC** — `CudaIPC` type, uses `reduce_tensor()` / `rebuild_cuda_tensor()`, zero-copy GPU (Linux only)
2. **Pool IPC** — `PoolIPC` type (**IN PROGRESS**, see below), uses `cudaMemPoolExportPointer`, zero-copy GPU
3. **PyTorch shared memory** — `TensorRef` type, uses `file_system` sharing strategy (/dev/shm), zero-copy CPU
4. **NumPy** — converted to PyTorch tensor, uses strategy 3
5. **Trimesh/Pickle** — pickled to `SharedMemory` block
6. **Primitives** — inline in JSON

### Supporting Files
- `src/comfy_env/isolation/tensor_utils.py` — `TensorKeeper` (prevents GC of shared tensors), `prepare_tensor_for_ipc()`, `release_tensor()` (madvise DONTNEED)
- `src/comfy_env/isolation/workers/subprocess.py` lines 275-280 — IPC handle forwarding cache (`_cuda_ipc_metadata_cache`) for zero-copy Worker A → Parent → Worker B chains
- `docs/worker-serialization.md` — Detailed docs on the serialization system

## Development

```bash
cd /home/shadeform/utils/comfy-env
pip install -e ".[dev]"
```

ComfyUI lives at `~/geometrypack/ComfyUI/` (or `~/ComfyUI/`). Workers find it via `COMFYUI_BASE` env var or by walking up from `working_dir`.

## Current Work: Shareable CUDA Memory Pool IPC

### The Problem

ComfyUI's `cuda_malloc.py` (at `~/geometrypack/ComfyUI/cuda_malloc.py`) sets:
```
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
```

This propagates to worker subprocesses and **completely breaks legacy CUDA IPC**:
- `reduce_tensor()` throws: `RuntimeError: cudaMallocAsync does not yet support shareIpcHandle`
- Both parent and worker `_probe_cuda_ipc()` functions are **buggy** — they test `Event` + tensor allocation but never test `reduce_tensor()`, so they return `True` even though IPC is broken
- Result: CUDA tensor serialization crashes, or silently falls back to CPU shared memory (2 copies)

### The Solution (approved, implementation in progress)

**Workers only** — create a shareable CUDA memory pool and use pool-based IPC instead of legacy IPC. No changes to the parent/ComfyUI process.

| Direction | Mechanism | Cost |
|-----------|-----------|------|
| Worker → Parent | `cudaMemPoolExportPointer` → `cudaMemPoolImportPointer` | Zero-copy |
| Worker → Worker | Pool handle forwarding via parent | Zero-copy (v2, CPU fallback in v1) |
| Parent → Worker | CPU shared memory (existing TensorRef path) | 2 copies |

#### Implementation Plan (10 phases)

1. **ctypes bindings** — Wrap 6 CUDA runtime functions: `cudaMemPoolCreate`, `cudaMemPoolExportToShareableHandle`, `cudaDeviceSetMemPool`, `cudaMemPoolExportPointer`, `cudaMemPoolImportPointer`, `cudaMemPoolImportFromShareableHandle`
2. **FD passing** — `SCM_RIGHTS` over the existing AF_UNIX socket for pool handle exchange
3. **Fix `_probe_cuda_ipc()`** — Add `reduce_tensor()` test on both parent and worker sides. Parent probe returns `False` under `cudaMallocAsync` (triggers CPU fallback, which is correct)
4. **Worker pool creation** — After torch import in worker script (~line 912), create shareable pool with `cudaMemHandleTypePosixFileDescriptor`, set as current device pool via `cudaDeviceSetMemPool`
5. **Pool FD exchange** — Worker exports pool FD after `{"status": "ready"}`, parent imports it in `_ensure_started()`
6. **Pool serialization** — New `PoolIPC` type in `_serialize_pool_ipc()` / `_deserialize_pool_ipc()` on both sides
7. **Wire into `_to_shm()` / `_from_shm()`** — Worker uses `PoolIPC` for CUDA tensors when pool available; parent handles `PoolIPC` in deserialization
8. **Worker patches** — `torch.cuda.empty_cache()` patched to also trim shareable pool (`cudaMemPoolTrimTo`)
9. **Documentation** — `docs/ipc.md`
10. **Testing** — End-to-end verification

#### Key Technical Details

- `cudaMemPoolExportPointer` produces a 64-byte opaque `cudaMemPoolPtrExportData` struct → base64-encoded in JSON
- Parent needs the worker's pool handle to import pointers → exchanged via FD passing at connection time
- `_from_shm()` is module-level but needs worker pool handle → use module-level `_active_worker_pool` set by `SubprocessWorker.call_method()`/`call_module()` before calling `_from_shm()`
- PyTorch's `cudaMallocAsync` backend uses `cudaMallocAsync(ptr, size, stream)` (3-arg form) which allocates from the **current** device pool — so `cudaDeviceSetMemPool` redirects all PyTorch allocations
- PyTorch's `empty_cache()` and `memory_stats()` query the **default** pool (bug in PyTorch's `CUDAMallocAsyncAllocator.cpp` — uses `cudaDeviceGetDefaultMemPool` instead of `cudaDeviceGetMemPool`). Worker patch handles `empty_cache`; `memory_stats` is left as-is (conservative undercount is safe)
- `tensor_utils.py` `prepare_tensor_for_ipc()` also needs `cudaMallocAsync` error handling

#### What NOT to Change
- **No monkey-patching ComfyUI** — all changes are in comfy-env workers
- **No changes to parent's allocator** — parent keeps `cudaMallocAsync`, uses CPU fallback for parent→worker
- **No changes to `SocketTransport`** — FD passing uses raw socket directly

### Related Discovery: pyisolate Has the Same Bug

`/home/shadeform/pyisolate/pyisolate/_internal/tensor_serializer.py` line 316 and `torch_utils.py` line 28-51 have identical bugs — `reduce_tensor()` call with incomplete error handling, probe that doesn't test `reduce_tensor()`.
