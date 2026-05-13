# comfy-env

Environment management and automatic CUDA wheel resolution/installation for ComfyUI custom nodes.

## The Problem(s) and Solution(s)

Problem 1:
ComfyUI custom nodes share a single Python environment. This breaks when:
- Node A needs torch 2.4, Node B needs torch 2.8
- Two packages bundle conflicting native libraries (libomp, CUDA runtimes, cv2)
- A node requires a specific Python version (e.g., Blender needs 3.11, pymesh2 needs 3.9)

Solution 1:
comfy-env provides **process isolation** -- nodes that need conflicting dependencies run in their own Python environments as persistent subprocesses, transparent to ComfyUI.

Two config files, two roles:

- **`comfy-env-root.toml`** (root level): System packages (apt/brew) and ComfyUI node dependency management. Never touches the Python environment -- PyPI deps stay in `requirements.txt`.
- **`comfy-env.toml`** (subdirectories): Each subdirectory with this file gets its own isolated Python environment via [pixi](https://pixi.sh), with a separate interpreter, conda packages, pip packages, and pre-built CUDA wheels.

Using conda CANNOT be avoided. We need to be able to use ComfyUI functions and return types within isolated nodes, which means they will need things like `av` or `ffmpeg`, and those cannot be installed through PyPI. [Pixi](https://pixi.sh) is a convenient, fast, Rust-based package manager that speaks both conda-forge AND PyPI in the same `pixi.toml`, ships a real lockfile (so envs are reproducible across machines), installs entirely per-user with no admin / no system Python pollution, and uses [uv](https://github.com/astral-sh/uv) under the hood for the PyPI side -- so it's as fast as the fastest thing in the ecosystem.

Problem 2:
With the advent of ever more complex and useful Computer Vision ML models, code relies on CUDA packages like flash-attn, nvdiffrast, nunchaku, pytorch3d to work.
Every single CUDA compiled wheel is compiled for:
- Python ABI (3.10, 3.11, 3.12 etc)
- Pytorch version when linking against pytorch, can be from 2.4 to 2.11
- CUDA version (12.8, 13.0)
- OS (Windows vs Linux)
- GPU architectures (8.0 and above)

Solution 2:
In order to ensure a smooth use and installation for ComfyUI users, we offer automatic wheel resolution through a cuda index: https://github.com/PozzettiAndrea/cuda-wheels.

## Architecture

```
ComfyUI-MyPack/
+-- comfy-env-root.toml            # System packages + node deps
+-- install.py                     # from comfy_env import install; install()
+-- prestartup_script.py           # from comfy_env import setup_env; setup_env()
+-- __init__.py                    # from comfy_env import register_nodes
`-- nodes/
    +-- main/                      # No config -> imported in main process
    |   `-- __init__.py
    `-- cgal/                      # Has config -> isolated subprocess
        +-- comfy-env.toml
        `-- __init__.py

<workspace>/                       # %LOCALAPPDATA%\Programs\comfy-env  (Windows)
                                   # ~/.ce                              (Unix)
+-- pixi.toml                      # Generated from every discovered comfy-env.toml
`-- .pixi/envs/
    +-- mypack-cgal/               # Env for ComfyUI-MyPack/nodes/cgal/
    |   +-- python                 #   Complete Python interpreter
    |   `-- lib/.../site-packages/ #   + isolated packages
    `-- <plugin>-<subdir>/         # Each isolation config gets its own named env
        `-- ...
```

Env names are derived as `<plugin>-<subdir>` (or just `<plugin>` for root-level configs), with the `ComfyUI-` prefix stripped and the result lowercased: `comfyui-sam3/nodes` -> `sam3-nodes`, `comfyui-motioncapture/nodes` -> `motioncapture-nodes`, etc. See [`get_env_name`](src/comfy_env/environment/cache.py).

### Examples in the wild

- **[ComfyUI-TRELLIS2](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2)** -- root-only config (`comfy-env-root.toml`). No isolation env; uses comfy-env purely for CUDA wheel resolution (`flash-attn`, `sageattention`) and to declare `ComfyUI-GeometryPack` as a node dependency. Runs inside the host ComfyUI process.
- **[ComfyUI-Hunyuan3D-Part](https://github.com/PozzettiAndrea/ComfyUI-Hunyuan3D-Part)** -- same pattern; lightest possible use of comfy-env.
- **[ComfyUI-GeometryPack](https://github.com/PozzettiAndrea/ComfyUI-GeometryPack)** -- full isolation env. `nodes/comfy-env.toml` declares conda deps from `conda-forge` + a custom `pozzettiandrea` channel (CGAL, igl, `bpy`, pyvista), one CUDA package (`cumesh`), PyPI deps pinned via custom simple indexes (`pyQuadriFlow`, `pygeogram`, `pypmp`, `pymesh2`), and per-platform extras (`mesalib`/`libglu`/`xorg-libsm` on Linux, `embreex`/`msvc-runtime` on Windows). This is the env that doesn't fit in the host venv.
- **[cookiecutter-comfy-extension](https://github.com/PozzettiAndrea/cookiecutter-comfy-extension)** -- scaffold for new node packs; ships a minimal `comfy-env.toml` template and the canonical `install.py` / `prestartup_script.py` / `__init__.py` triplet.



## How It Works

**Build time** (`install.py`): For each subdirectory with a `comfy-env.toml`, comfy-env computes its env name (see above), generates a single `pixi.toml` in the workspace with one `[environments.<name>]` entry per discovered config, and runs `pixi install -e <name>` to materialize each environment. Identical env names from different ComfyUI installs share the same materialized env on disk -- env names act as the global identifier.

**Workspace location**: a single per-user pixi workspace, shared by every ComfyUI install on the machine.
On Windows the default is `%LOCALAPPDATA%\Programs\comfy-env` (sits next to the ComfyUI Desktop install at `%LOCALAPPDATA%\Programs\ComfyUI`; never needs admin to create).
If comfy-env detects an env at the legacy drive-root path `C:\ce`, it prints a one-line "please reinstall" nudge at startup so users notice the migration.

On Linux/MACOS the default is `~/.ce`. Override with the `COMFY_ENV_ROOT` env var. 

**Runtime** (`register_nodes()`): Discovers all node subdirectories. Those with a built env at `<workspace>/.pixi/envs/<env_name>/` run in persistent subprocess workers using the isolated Python interpreter. Those without a config (or whose env hasn't been materialized yet) are imported normally in the main ComfyUI process. Workers communicate via Unix domain sockets (TCP on Windows) and support bidirectional callbacks for VRAM budget negotiation and progress reporting.

**CUDA packages**: Listed in `[cuda]`, installed from the PyTorch wheel index or [cuda-wheels](https://pozzettiandrea.github.io/cuda-wheels/) -- pre-built wheels for nvdiffrast, pytorch3d, gsplat, flash-attn, etc. No CUDA toolkit or C++ compiler needed. The resolver tries the GitHub Pages simple index first, retries transient TCP-reset errors with a real User-Agent (corp proxies / AV products RST `Python-urllib`), and falls back to the GitHub Releases API on a different routing edge when Pages is unreachable end-to-end.

## Startup logging

On every ComfyUI launch, `comfy-env`'s prestartup hook tells you exactly where envs live and whether each one is built:

```
[comfy-env] Workspace: C:\Users\you\AppData\Local\Programs\comfy-env
[comfy-env] comfyui-motioncapture: 1 isolation env(s):
[comfy-env]   nodes -> C:\Users\you\...\custom_nodes\comfyui-motioncapture\nodes
[comfy-env]     env: C:\Users\you\AppData\Local\Programs\comfy-env\.pixi\envs\motioncapture-nodes  [OK]
[comfy-env] prestartup complete
```

`[MISSING -- run install.py]` instead of `[OK]` means the config was discovered but the pixi env hasn't been materialized; run the node's `install.py` to build it.

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