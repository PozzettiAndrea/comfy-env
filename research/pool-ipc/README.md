# Zero-copy CUDA transfer under `backend:cudaMallocAsync`

Reproducible experiments behind
[ADR-0030](https://pozzettiandrea.github.io/comfy-forge-docs/comfy-env/adr/0030-gpu-floors-and-zero-copy-contract/)
and the [Zero-copy CUDA transfer](https://pozzettiandrea.github.io/comfy-forge-docs/comfy-env/zero-copy-ipc/)
page. Nothing here is imported by `comfy_env`; these are standalone
scripts that answer questions the docs assert answers to.

Run them on a machine with an NVIDIA GPU. Only `verify_pool_swap.py`
needs torch; the other two are ctypes against `libcuda.so.1` /
`nvcuda.dll` and run on a stock system Python 3.9+.

| script | question it answers |
|---|---|
| `verify_pool_swap.py` | Can a worker read and write a parent's torch tensors with no copy, while the parent keeps ComfyUI's default `cudaMallocAsync` allocator? |
| `repro_mempool_import_segv.py` | At what allocation size does `cuMemPoolImportPointer` start segfaulting the importer? (Minimal repro attached to the NVIDIA bug report.) |
| `wddm_pool_probe.py` | Does WIN32 memory-pool export work on a WDDM GeForce? **Not yet run — needs a Windows machine.** |

## What was measured

On an RTX 3090 (GA102, 24576 MiB), driver 580.126.20, Linux 6.8,
torch 2.8.0+cu128:

- Creating a pool with `handleTypes = POSIX_FILE_DESCRIPTOR` and making
  it current with `cuDeviceSetMemPool()` **before torch is imported**
  survives torch initialization. Torch keeps the `cudaMallocAsync`
  backend and every tensor it allocates comes from our pool (checked
  with `cuPointerGetAttribute`), so every tensor is exportable.
- A worker process imports the pool over `SCM_RIGHTS`, imports a
  pointer, reads the parent's tensor with no copy, **writes into it in
  place**, and the parent sees the write through its own
  `torch.Tensor`. Interprocess events work. Teardown in the mandated
  importer-frees-first order works.
- Pointer import costs about 1.25 ms/GiB and is linear.
- `torch.cuda.memory_allocated()` and `memory_reserved()` read 0,
  because torch's statistics query the device's *default* pool while
  allocations come from ours. ComfyUI is unaffected: it computes free
  VRAM as `mem_get_info` plus a `reserved - active` correction, and the
  correction term goes to zero rather than wrong.

### The undocumented ordering requirement

`cuMemPoolExportPointer()` returns `CUDA_ERROR_INVALID_VALUE` for every
allocation until `cuMemPoolExportToShareableHandle()` has been called on
that pool. Export the pool's handle first, once, at startup.

### The driver bug

`cuMemPoolImportPointer()` **segfaults the importing process** for any
single allocation larger than 5248 MiB. Bisected at 16 MiB granularity:
5248 MiB imports in 6.6 ms, 5264 MiB dies with `SIGSEGV`,
`si_addr=0xd4` — a NULL dereference at a struct offset, three frames
below `cuMemPoolImportPointer` inside `libcuda`. Not host OOM, not
device OOM, not stack exhaustion (tested with a 512 MiB thread stack and
`ulimit -s unlimited`), not a 2^32 boundary, and every NVIDIA ioctl
returns success up to the fault.

The limit is **per allocation, not cumulative**: four separate 4096 MiB
allocations from the same pool (16 GiB total) all import successfully.
Any transfer built from sub-5 GiB tensors is unaffected.

No prior report of this exists — searched the NVIDIA developer forums
(zero topics for the API name), GitHub globally (eight issues mention
the API, none about crashes), the 570/575/580 driver release notes, and
the issue trackers of pytorch, CuPy, JAX, TensorFlow, vLLM, SGLang,
NCCL, Numba and Ray. NVIDIA's own `streamOrderedAllocationIPC` sample
uses a 64 MiB buffer, two orders of magnitude below the threshold,
which is the likely reason it went unnoticed.

`NVIDIA-BUG-REPORT.md` is the writeup, ready to file.
