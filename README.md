# comfy-env

Environment management for ComfyUI custom nodes.

## The Problem

ComfyUI custom nodes share a single Python environment. This breaks when:
- Node A needs torch 2.4, Node B needs torch 2.8
- Two packages bundle conflicting native libraries (libomp, CUDA runtimes)
- A node requires a specific Python version (e.g., Blender API needs 3.11)

## The Solution

comfy-env provides **process isolation** — nodes that need conflicting dependencies run in their own Python environments as persistent subprocesses, transparent to ComfyUI.

Two config files, two roles:

- **`comfy-env-root.toml`** (root level): System packages (apt/brew) and ComfyUI node dependency management. Never touches the Python environment — PyPI deps stay in `requirements.txt`.
- **`comfy-env.toml`** (subdirectories): Each subdirectory with this file gets its own isolated Python environment via [pixi](https://pixi.sh), with a separate interpreter, conda packages, pip packages, and pre-built CUDA wheels.

## Architecture

```
ComfyUI-MyPack/
├── comfy-env-root.toml            # System packages + node deps
├── install.py                     # from comfy_env import install; install()
├── prestartup_script.py           # from comfy_env import setup_env; setup_env()
├── __init__.py                    # from comfy_env import register_nodes
└── nodes/
    ├── main/                      # No config → imported in main process
    │   └── __init__.py
    └── cgal/                      # Has config → isolated subprocess
        ├── comfy-env.toml
        ├── _env_a1b2c3 ─────────► ~/.ce/_env_a1b2c3/.pixi/envs/default
        └── __init__.py

~/.ce/                             # Central build cache (C:/ce on Windows)
└── _env_a1b2c3/                   # SHA256(config + comfy-env version)[:8]
    ├── .pixi/envs/default/        # Complete Python environment
    │   ├── bin/python
    │   └── lib/python3.11/site-packages/
    ├── pixi.toml                  # Generated from comfy-env.toml
    └── .done                      # Build complete marker
```

## How It Works

**Build time** (`install.py`): For each subdirectory with a `comfy-env.toml`, comfy-env hashes the config contents + its own package version, checks the central cache (`~/.ce`), and builds a pixi environment if needed. The result is linked into the node directory as `_env_<hash>` — symlink on Unix, NTFS junction on Windows (no admin required). Identical configs across different node packs share the same cached build.

**Runtime** (`register_nodes()`): Discovers all node subdirectories. Those with a built `_env_*` run in persistent subprocess workers using the isolated Python interpreter. Those without a config are imported normally. Workers communicate via Unix domain sockets (TCP on Windows) and support bidirectional callbacks for VRAM budget negotiation and progress reporting.

**CUDA packages**: Listed in `[cuda]`, installed from the PyTorch wheel index or [cuda-wheels](https://pozzettiandrea.github.io/cuda-wheels/) — pre-built wheels for nvdiffrast, pytorch3d, gsplat, etc. No CUDA toolkit or C++ compiler needed.

## Usage

```python
# install.py
from comfy_env import install
install()

# prestartup_script.py
from comfy_env import setup_env
setup_env()

# __init__.py
from comfy_env import register_nodes
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()
```

```bash
pip install comfy-env
```

## Docs

- [Getting Started](docs/getting-started.md) — setup guide and walkthrough
- [Config Reference](docs/config-reference.md) — all config options for both files
- [CLI](docs/cli.md) — command-line tools
- [Build Internals](docs/isolation-build.md) — how isolated environments are built

## Example

[ComfyUI-GeometryPack](https://github.com/PozzettiAndrea/ComfyUI-GeometryPack) — multiple isolated environments (CGAL, Blender, GPU) with per-subdirectory configs and different Python versions.
