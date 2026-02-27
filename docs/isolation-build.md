# How Isolated Environments Are Built

This documents the internals of how comfy-env creates and manages isolated Python environments for ComfyUI custom nodes.

## Overview

Nodes that need conflicting dependencies (different Python versions, incompatible libraries) run in **isolated environments** — separate pixi (conda+pip) environments with their own Python interpreter. Each subdirectory with a `comfy-env.toml` gets its own environment.

Environments are built once in a central cache (`~/.ce` on Unix, `C:/ce` on Windows) and linked into node directories as `_env_<hash>`. On Windows this uses NTFS junctions (no admin/Developer Mode required); on Unix, symlinks.

## Config Hash & Env Identity

**File**: `environment/cache.py` — `compute_config_hash()`

Each environment is identified by a hash of:
- The `comfy-env.toml` file contents
- The comfy-env package version

```
hash = SHA256(config_bytes + version_string)[:6]
```

This means:
- Changing the config creates a new environment
- Upgrading comfy-env invalidates all caches
- Two nodes with identical `comfy-env.toml` share the same cached build

## Build Directory Structure

```
~/.ce/                              # Central build cache
├── _env_a1b2c3/                    # One per unique config hash
│   ├── .done                       # Present = build complete
│   ├── .building/                  # Lock dir (atomic mkdir)
│   ├── install.log                 # Full build output
│   ├── .comfy-env-meta.json        # Source node name, config content
│   ├── pixi.toml                   # Generated from comfy-env.toml
│   └── .pixi/envs/default/        # The actual conda environment
│       ├── bin/python              # (Unix)
│       ├── python.exe              # (Windows)
│       ├── lib/python3.11/site-packages/
│       └── lib/                    # Shared libraries
├── detect.sh                       # Lists all envs + status
└── detect.bat
```

In the node directory:
```
nodes/cgal/_env_a1b2c3  ->  ~/.ce/_env_a1b2c3/.pixi/envs/default
```

## Build Steps

Traced from `_install_via_pixi()` in `install.py`:

### 1. Hash & paths

Compute config hash, determine:
- `build_dir` = `~/.ce/_env_<hash>`
- `env_path` = `<node_dir>/_env_<hash>` (the link that will point to the build)

### 2. Fast path (reuse)

If `build_dir/.done` exists, the env is already built. Just create the link from `env_path` to `build_dir/.pixi/envs/default` and return.

### 3. Build lock

Acquire lock via atomic `mkdir build_dir/.building/`:
- If `mkdir` succeeds: we own the build, proceed
- If `FileExistsError`: another process is building — poll for `.done` every 1s, up to 10 minutes
- If timeout: assume stale lock from a crashed build, nuke `build_dir` and rebuild

### 4. Bootstrap pixi

`ensure_pixi()` in `packages/pixi.py`:
1. Check PATH, venv bin dir, `~/.pixi/bin/`, `~/.local/bin/`
2. If not found: download from GitHub releases (platform-specific binary)
3. Make executable

### 5. Generate pixi.toml

`write_pixi_toml()` in `packages/toml_generator.py`:
- Sets project name, version, channels (`conda-forge`)
- Pins Python version from config (or current interpreter)
- Passes through `[dependencies]` (conda) and `[pypi-dependencies]` (pip) from config
- Adds system requirements: `glibc = 2.35` on Linux, `cuda` major version if GPU detected
- Adds PyTorch packages from `[cuda]` to `[pypi-dependencies]` with per-package `index` pointing to the correct PyTorch wheel index:
  - GPU: `torch = { version = "==2.8.*", index = "https://download.pytorch.org/whl/cu128" }`
  - No GPU: same but with `.../cpu`
- Configures system requirements: `glibc = 2.35` on Linux, `cuda` major version if GPU detected

### 6. pixi install

Runs `pixi install` in the build directory. This resolves all conda, pip, **and PyTorch** dependencies in a single pass and creates `.pixi/envs/default/` with a complete Python environment. PyTorch packages are resolved from the per-package index URL, ensuring the correct CUDA variant is installed alongside all other dependencies without conflicts.

Environment variables are set to prevent pixi from downloading its own Python:
```
UV_PYTHON_INSTALL_DIR = build_dir/_no_python
UV_PYTHON_PREFERENCE = only-system
```

### 7. CUDA wheels installation

For **non-PyTorch** packages listed in `[cuda]` (nvdiffrast, pytorch3d, gsplat, etc.), installed into the pixi env via `uv pip install` after pixi:

- Resolved from the [cuda-wheels index](https://pozzettiandrea.github.io/cuda-wheels/)
- Matched by: package name, CUDA version, torch version, Python version, platform
- Installed with `--no-deps --no-cache` to avoid conflicts
- Falls back to PyPI if not in the index

PyTorch packages (torch, torchvision, torchaudio) are **not** installed in this step — they are handled by pixi in step 6.

### 8. Link & finalize

1. Create link: `_env_<hash>` -> `build_dir/.pixi/envs/default`
2. Clean up any `.pixi/` dir in the node directory
3. Touch `.done` marker
4. Save metadata (node name, config content, summary)
5. Release lock (`rmdir .building/`)

## GPU & CUDA Detection

**Files**: `detection/cuda.py`, `detection/gpu.py`

Detection order for GPU info:
1. **NVML** (pynvml) — compute capability, VRAM, driver version
2. **nvidia-smi** — subprocess, parse output
3. **PyTorch** — `torch.cuda.get_device_properties()`
4. **sysfs** — scan `/sys/bus/pci/devices` for NVIDIA vendor ID (Linux only)

On Windows, nvidia-smi is tried first (avoids DLL loading issues with NVML).

CUDA version detection:
1. `COMFY_ENV_CUDA_VERSION` env var (manual override)
2. `torch.version.cuda` (from installed PyTorch)
3. `nvcc --version` output

Version mapping (in `cuda_wheels.py`):
```
CUDA 12.8 -> PyTorch 2.8
CUDA 12.4 -> PyTorch 2.4
```

Blackwell GPUs (compute capability 10.x) are forced to CUDA 12.8.

## Runtime: How Isolation Envs Are Used

**File**: `isolation/wrap.py`, `isolation/metadata.py`

### Node registration

`register_nodes()` discovers all `comfy-env.toml` files under the node package:
1. Find `_env_*` directories next to each config
2. For each isolation env: spawn a subprocess in the env's Python to import the node module and extract metadata (INPUT_TYPES, RETURN_TYPES, FUNCTION, etc.)
3. Build proxy classes from the metadata — these have all the ComfyUI attributes but delegate execution to a subprocess worker

### Worker pool

One persistent `SubprocessWorker` per isolation env, reused across calls:
- Workers communicate via Unix domain sockets (Linux/macOS) or TCP localhost (Windows)
- Auto-restart on crash (native segfault, etc.)
- Cleaned up via `atexit` handler

### Environment isolation

`build_isolation_env()` sets platform-specific environment for the subprocess:
- **Linux**: `LD_LIBRARY_PATH` = env lib dir + system lib dirs
- **macOS**: `DYLD_LIBRARY_PATH` = env lib dir
- **Windows**: `PATH` = env root + Scripts + Library/bin + system dirs; `KMP_DUPLICATE_LIB_OK=TRUE`

### Bidirectional IPC

Workers support callbacks from subprocess to parent:
- **VRAM budget**: subprocess requests GPU memory, parent evicts its own models to make room
- **Progress**: subprocess reports progress bar updates to ComfyUI frontend

## Concurrency & Caching

- **Build lock**: atomic `mkdir` prevents parallel builds of the same config hash
- **Config hash**: changing the config or upgrading comfy-env creates new envs; old ones remain in `~/.ce/` until manual deletion
- **Metadata cache**: subprocess node scans are cached in `_env_*/.metadata_cache.pkl`, invalidated when any `.py` file in the package changes (based on mtime hash)
