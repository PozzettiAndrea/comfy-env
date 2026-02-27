# Worker Lifecycle

**Files**: `isolation/workers/subprocess.py`, `isolation/wrap.py`

## Startup

### 1. Socket creation

The parent creates a listening socket:
- **Linux/macOS**: Unix domain socket at `/dev/shm/comfy_worker_*.sock` (or temp dir)
- **Windows**: TCP on `127.0.0.1:<random_port>`

### 2. Process launch

The worker script is written to a temp directory and launched:

```bash
python /tmp/comfyui_pvenv_*/persistent_worker.py <socket_addr>
```

Environment is configured by `build_isolation_env()`:
- **Linux**: `LD_LIBRARY_PATH` = env lib dir + system lib dirs
- **macOS**: `DYLD_LIBRARY_PATH` = env lib dir
- **Windows**: `PATH` = env root + Scripts + Library/bin + system dirs

stderr is inherited (not piped) to avoid deadlocks from tqdm/progress bars.

### 3. Handshake

```
Parent                          Worker
  │                               │
  │── accept() (60s timeout)      │── connect()
  │                               │
  │── {sys_paths: [...]} ────►    │── add sys_paths
  │                               │── redirect print() to socket
  │                               │── hook nn.Module.to() for model detection
  │                               │── shim load_models_gpu()
  │                               │── install progress bar hook
  │   ◄── {status: ready}        │
```

After the handshake, the worker enters its message loop and waits for requests.

## Health Check

Periodic ping to verify the worker is responsive:

```python
# Parent sends:
{"method": "ping"}

# Worker responds:
{"status": "pong"}
```

Timeout is configurable via `[options] health_check_timeout` in `comfy-env.toml` (default: 5 seconds).

## Request Handling

For each node execution:

1. **Parent serializes inputs** to shared memory (`_to_shm()`)
2. **Parent sends request** via socket:
   ```python
   {"type": "call_method", "module": "...", "class_name": "...",
    "method_name": "...", "kwargs": {...}, "self_state": {...}}
   ```
3. **Parent enters receive loop**, handling intermediate messages:
   - `{type: log}` → print to stderr
   - `{type: callback}` → dispatch (VRAM budget, progress), send response
   - `{status: ok/error}` → break loop
4. **Parent deserializes result** from shared memory (`_from_shm()`)
5. **Parent cleans up** shared memory blocks

## Progress Reporting

The worker hooks into `comfy.utils.set_progress_bar_global_hook()`:

```
Worker                              Parent                    Frontend
  │                                   │                         │
  │── ProgressBar(total=10)           │                         │
  │── step 1                          │                         │
  │── callback: report_progress ──►   │                         │
  │   value=1, total=10               │── PROGRESS_BAR_HOOK ──► │
  │   ◄── callback_response           │                         │
  │                                   │                         │
  │── step 2                          │                         │
  │── callback: report_progress ──►   │                         │
  │   value=2, total=10               │── check interrupted?    │
  │                                   │   (user clicked cancel) │
  │   ◄── {status: error,             │                         │
  │        error: "interrupted"}      │                         │
  │── raise InterruptedError          │                         │
```

The parent checks `comfy.model_management.throw_exception_if_processing_interrupted()` on each progress callback. If the user cancelled, the error propagates back to the worker which stops processing.

## Crash Recovery

Workers auto-restart on crash. Detection happens before each request:

1. Check `process.poll()` — is the process still alive?
2. Check `_check_socket_health()` — does ping respond?
3. If either fails:
   - Kill the process
   - Fire `_on_restart()` → clean up stale model patchers
   - Respawn from socket creation step

### Crash Diagnostics

On crash, the parent prints:
- Exit code and signal name
- Last 20 lines of worker debug log (`/tmp/comfy_worker_debug.log`)
- Last 20 lines of faulthandler dump (`/tmp/comfy_worker_faulthandler.log`)

Debug logging can be enabled with `COMFY_ENV_DEBUG=1`.

## Error Propagation

Exceptions in the worker are caught, serialized, and re-raised in the parent:

```python
# Worker sends:
{"status": "error", "error": "ValueError: bad input", "traceback": "..."}

# Parent raises:
WorkerError("ValueError: bad input", traceback="...")
```

`WorkerError` includes the worker's full traceback for debugging.

## Shutdown

Triggered by `atexit` handler or explicit cleanup:

1. Send `{"method": "shutdown"}` to worker
2. Close socket and server socket
3. Remove Unix socket file (if applicable)
4. `process.wait(timeout=5)` — if timeout, `kill()`
5. Remove temp directory containing worker script

## Stale Worker Cleanup

On startup, `register_nodes()` cleans up orphaned workers from previous runs:
- Scans for stale Unix socket files in `/dev/shm/`
- Removes stale temp directories matching `comfyui_pvenv_*`
- Ensures clean state before spawning new workers
