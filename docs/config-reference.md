# Config Reference

comfy-env uses two config files with distinct roles:

| File | Location | Role |
|------|----------|------|
| `comfy-env-root.toml` | Node pack root | System packages, node dependencies |
| `comfy-env.toml` | Subdirectories | Isolated Python environment |

## comfy-env-root.toml

Root-level config. Handles system packages and ComfyUI node dependencies. **Does not install any Python packages**.

```toml
[apt]
packages = ["libgl1-mesa-glx", "libglib2.0-0"]

[brew]
packages = ["libomp"]

[node_reqs]
ComfyUI_essentials = { github = "cubiq/ComfyUI_essentials", tag = "v1.0.0" }
comfyui_controlnet_aux = "Fannovel16/comfyui_controlnet_aux"
```

### `[apt]`

Linux system packages installed via `apt-get`.

| Field | Type | Description |
|-------|------|-------------|
| `packages` | list of strings | Package names to install |

### `[brew]`

macOS packages installed via Homebrew.

| Field | Type | Description |
|-------|------|-------------|
| `packages` | list of strings | Package names to install |

### `[node_reqs]`

ComfyUI custom node dependencies. Each entry is installed into ComfyUI's `custom_nodes/` directory, then its `requirements.txt` and `install.py` are run. Dependencies are resolved recursively.

Three source types:

```toml
[node_reqs]
# GitHub — default branch
ComfyUI_essentials = "cubiq/ComfyUI_essentials"

# GitHub — pinned to tag
ComfyUI_essentials = { github = "cubiq/ComfyUI_essentials", tag = "v1.0.0" }

# GitHub — pinned to branch
ComfyUI_dev = { github = "owner/repo", branch = "dev" }

# GitHub — pinned to commit
ComfyUI_pinned = { github = "owner/repo", commit = "abc123def456" }

# ComfyUI Registry — specific version
comfyui_essentials = { registry = "comfyui_essentials", version = "1.0.0" }

# ComfyUI Registry — latest
comfyui_essentials = { registry = "comfyui_essentials" }
```

| Source | Fields | Description |
|--------|--------|-------------|
| GitHub (string) | `"owner/repo"` | Clone default branch (shallow) |
| GitHub (dict) | `github`, optional `tag`/`branch`/`commit` | Clone with pinned ref |
| Registry (dict) | `registry`, optional `version` | Download from [registry.comfy.org](https://registry.comfy.org) |

After installing node dependencies, comfy-env re-installs the main package's `requirements.txt` to restore any versions that got overwritten.

---

## comfy-env.toml

Per-subdirectory config. Each subdirectory with this file gets its own isolated Python environment built via [pixi](https://pixi.sh).

```toml
python = "3.11"

[dependencies]
cgal = "*"

[pypi-dependencies]
trimesh = { version = "*", extras = ["easy"] }

[cuda]
packages = ["torch", "torchvision", "nvdiffrast"]

[env_vars]
OMP_NUM_THREADS = "4"

[options]
health_check_timeout = 10.0
```

### `python`

Python version for the isolated environment.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `python` | string | Current interpreter version | e.g., `"3.11"`, `"3.12"` |

### `[dependencies]`

Conda packages installed via pixi (from conda-forge).

```toml
[dependencies]
cgal = "*"
boost = ">=1.80"
```

Uses [pixi dependency syntax](https://pixi.sh/latest/reference/pixi_manifest/#dependencies).

### `[pypi-dependencies]`

PyPI packages installed via pixi's pip integration.

```toml
[pypi-dependencies]
trimesh = { version = "*", extras = ["easy"] }
numpy = "*"
```

Uses [pixi PyPI dependency syntax](https://pixi.sh/latest/reference/pixi_manifest/#pypi-dependencies).

### `[cuda]`

Packages that need CUDA-aware resolution. comfy-env automatically detects the GPU and CUDA version, then resolves these packages accordingly.

| Field | Type | Description |
|-------|------|-------------|
| `packages` | list of strings | Package names (e.g., `torch`, `torchvision`, `nvdiffrast`) |

**PyTorch packages** (torch, torchvision, torchaudio) are added to pixi's `[pypi-dependencies]` with a per-package `index` URL pointing to the correct PyTorch wheel index (e.g., `https://download.pytorch.org/whl/cu128`). This means they're resolved alongside all other dependencies in a single pixi pass, avoiding version conflicts.

**Custom CUDA packages** (nvdiffrast, pytorch3d, gsplat, etc.) are resolved from [cuda-wheels](https://pozzettiandrea.github.io/cuda-wheels/) — pre-built wheels matched by CUDA version, torch version, Python version, and platform. Installed via `uv pip install` after pixi. Falls back to PyPI if not in the index.

### `[env_vars]`

Environment variables set in the subprocess worker.

```toml
[env_vars]
OMP_NUM_THREADS = "4"
SOME_FLAG = "true"
```

These are applied to the isolated subprocess environment only, not the main ComfyUI process.

### `[options]`

Runtime options.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `health_check_timeout` | float | `5.0` | Seconds to wait for subprocess worker to start |

---

## Section Reference

| Section | Root | Isolated |
|---------|:----:|:--------:|
| `[apt]` | x | |
| `[brew]` | x | |
| `[node_reqs]` | x | |
| `python` | | x |
| `[dependencies]` | | x |
| `[pypi-dependencies]` | | x |
| `[cuda]` | | x |
| `[env_vars]` | | x |
| `[options]` | | x |

Any unrecognized TOML keys in `comfy-env.toml` are passed through directly to the generated `pixi.toml`.
