# Worker Serialization

**File**: `isolation/workers/subprocess.py`

JSON messages on the socket carry metadata only. Actual data (tensors, arrays, meshes) is transferred via **shared memory** — the JSON contains pointers that the receiving side uses to reconstruct the objects.

## Serialization Strategies

Inputs and outputs are recursively walked by `_to_shm()` / `_from_shm()`. Each object is serialized by the first matching strategy:

| Priority | Type | Strategy | Zero-copy? |
|----------|------|----------|------------|
| 1 | CUDA tensor | CUDA IPC handles | Yes (Linux only) |
| 2 | CPU tensor | PyTorch file_system shared memory | Yes |
| 3 | NumPy array | Convert to tensor → strategy 2 | Yes |
| 4 | Trimesh mesh | pickle → shared memory | No (copy) |
| 5 | SparseTensor | Decompose to coords + feats tensors | Per-tensor |
| 6 | Any pickleable | pickle → shared memory | No (copy) |
| 7 | Primitives | Inline in JSON | N/A |

## CUDA IPC (GPU tensors, Linux)

Uses `torch.multiprocessing.reductions.reduce_tensor()` to get an IPC handle — a file descriptor that lets another process map the same GPU memory.

```python
# Metadata in JSON
{
    "__type__": "CudaIPC",
    "tensor_size": [1, 3, 512, 512],
    "tensor_stride": [786432, 262144, 512, 1],
    "dtype": "torch.float32",
    "device_idx": 0,
    "handle": "<base64 IPC handle>",
    "storage_size": 786432,
    "event_handle": "<base64>",  # synchronization
}
```

The receiving side calls `rebuild_cuda_tensor()` to map the same GPU memory. No data is copied.

**IPC handle forwarding**: PyTorch's `reduce_tensor()` refuses to create a new IPC handle from a tensor received via IPC ("received from another process"). comfy-env solves this with **handle forwarding** — the original IPC metadata is cached on deserialization and forwarded directly when re-serializing to another worker. This enables true zero-copy for multi-hop chains (Worker A → Parent → Worker B) without cloning. Cloning only happens as a fallback when no cached metadata is available.

## PyTorch Shared Memory (CPU tensors)

Uses the `file_system` sharing strategy: the tensor's storage is memory-mapped to a file in `/dev/shm` (or temp directory).

```python
# Metadata in JSON
{
    "__type__": "TensorRef",
    "strategy": "file_system",
    "manager_path": "/dev/shm/...",
    "storage_key": "...",
    "storage_size": 786432,
    "dtype": "torch.float32",
    "tensor_size": [1, 3, 512, 512],
    "tensor_stride": [786432, 262144, 512, 1],
    "tensor_offset": 0,
}
```

The receiving side calls `rebuild_storage_filename()` to map the same shared memory file. No data is copied.

## NumPy Arrays

Converted to PyTorch tensors and serialized via strategy 2. A `__was_numpy__` flag is set so the receiving side converts back to NumPy.

Fallback: if PyTorch is unavailable, uses `multiprocessing.shared_memory.SharedMemory` with a byte copy.

## Trimesh / Pickle Fallback

Objects that can't use shared memory tensors are pickled and written to a `SharedMemory` block:

```python
# Metadata in JSON
{
    "__shm_trimesh__": True,  # or "__shm_pickle__": True
    "name": "shm_block_name",
    "size": 12345,
}
```

For Trimesh specifically, unpickleable native extensions (ray tracer, proximity query) are stripped before pickling.

## Object References

Some objects are too complex or stateful to serialize. These stay in the worker's memory and a reference ID is passed instead:

```python
# Metadata in JSON
{
    "__comfy_ref__": "ref_abc123",
    "__class__": "SomeComplexObject",
}
```

When the parent passes this reference back in a later call, the worker resolves it to the original object.

## Recursive Traversal

`_to_shm()` recursively walks dicts, lists, and tuples. Cycle detection prevents infinite loops (via `id()` tracking). A `registry` list accumulates all `SharedMemory` blocks created during serialization — these are cleaned up after the response is processed.

## Tensor Lifecycle (GC Prevention)

**Problem**: shared memory becomes invalid when the source tensor is garbage collected.

**Solution**: `TensorKeeper` (in `tensor_utils.py`) holds references to serialized tensors for 30-60 seconds after IPC transfer. Both parent and worker maintain their own keeper.

```python
class _TensorKeeper:
    def __init__(self, retention_seconds=60.0):
        self._keeper = deque()  # (timestamp, tensor_ref)

    def keep(self, t):
        # Add to queue, prune entries older than retention_seconds
```

## Shared Memory Cleanup

After each request-response cycle:
1. Parent calls `_cleanup_shm(registry)` — closes and unlinks all blocks created for the request
2. Worker cleans up input blocks before starting the next request
3. `ShmKeeper` on the worker side delays unlinking by 30s to avoid races
