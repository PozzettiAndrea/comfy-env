# Worker Memory Management

**Files**: `isolation/workers/subprocess.py`, `isolation/model_patcher.py`, `isolation/wrap.py`

The parent and subprocess workers share a single GPU. ComfyUI's memory manager only sees the parent's models. Without coordination, both processes would fight over VRAM. comfy-env solves this with **VRAM budget negotiation** and **cross-process model eviction**.

## Model Auto-Detection

When a subprocess worker moves a `torch.nn.Module` to CUDA (via `.to("cuda")`, `.cuda()`, etc.), comfy-env auto-detects it by hooking `nn.Module.to()`:

```python
# Worker-side hooks (subprocess.py)
# Intercepts: model.to("cuda"), model.cuda()
# Registers: model_id, size (params + buffers), kind
```

The model info is sent back to the parent in every response:

```python
{
    "status": "ok",
    "result": {...},
    "_new_models": [
        {"id": "model_abc", "size": 2147483648, "kind": "unet"},
    ]
}
```

## SubprocessModelPatcher

**File**: `isolation/model_patcher.py`

For each auto-detected model, the parent creates a `SubprocessModelPatcher` — a shim that exposes the model to ComfyUI's memory manager without the parent actually holding the model.

```
ComfyUI Memory Manager
  │
  ├── Real ModelPatcher (parent's own models)
  │
  └── SubprocessModelPatcher (proxy for worker's model)
        │
        ├── patch_model()   → IPC: model_to_device("cuda")
        └── unpatch_model() → IPC: model_to_device("cpu")
```

When ComfyUI needs to free VRAM, it calls `unpatch_model()` on the least-recently-used model. If that model is a subprocess model, the patcher sends an IPC command to move it to CPU.

## VRAM Budget Negotiation

When a subprocess node needs to load a model to GPU, it requests a VRAM budget from the parent:

```
Worker                                  Parent
  │                                       │
  │── load_models_gpu(models) ──►         │
  │   (intercepted by shim)               │
  │                                       │
  │── callback: request_vram_budget ──►   │
  │   model_info: [{size, kind}, ...]     │
  │   total_size: 2147483648              │
  │                                       │── free_memory(total * 1.1)
  │                                       │   (evicts parent models if needed)
  │                                       │
  │   ◄── callback_response               │
  │       device: "cuda:0"                │
  │       extra_reserved_vram: ...        │
  │       vram_state: "NORMAL"            │
  │                                       │
  │── real load_models_gpu() ──►          │
  │   (sees freed VRAM)                   │
```

The 1.1x multiplier provides 10% headroom to avoid OOM during the actual load.

## IPC Device Move

The parent can move a worker's model between devices:

```python
# Parent sends:
{"method": "model_to_device", "model_id": "model_abc", "device": "cpu"}

# Worker responds:
{"status": "ok", "device": "cpu", "moved": True}
```

This is triggered by ComfyUI's memory manager when it needs to reclaim VRAM for a different node's computation.

## Flow Summary

1. **Worker loads model to GPU** → auto-detected, reported to parent
2. **Parent creates SubprocessModelPatcher** → registered with ComfyUI's memory manager
3. **Another node needs VRAM** → ComfyUI evicts least-recently-used models
4. **If evicted model is subprocess model** → parent sends `model_to_device("cpu")` via IPC
5. **Worker runs a node that needs VRAM** → shim calls `request_vram_budget` callback
6. **Parent frees memory** → evicts its own models, returns budget
7. **Worker's `load_models_gpu()` runs** → sees freed VRAM, loads successfully
