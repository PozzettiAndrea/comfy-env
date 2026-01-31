# comfy-env

Environment management for ComfyUI custom nodes.

## Quick Start

```bash
pip install comfy-env
```

**1. Create `comfy-env.toml` in your node directory:**

```toml
[cuda]
packages = ["nvdiffrast", "pytorch3d"]

[pypi-dependencies]
trimesh = { version = "*", extras = ["easy"] }
```

**2. In `install.py`:**

```python
from comfy_env import install
install()
```

**3. In `prestartup_script.py`:**

```python
from comfy_env import setup_env
setup_env()
```

That's it. CUDA wheels install without compilation, and the environment is ready.

---

## Configuration

Create `comfy-env.toml` in your node directory:

```toml
# Python version for isolated environment (optional)
python = "3.11"

# CUDA packages from cuda-wheels index (no compilation needed)
[cuda]
packages = ["nvdiffrast", "pytorch3d", "flash-attn"]

# System packages (Linux only)
[apt]
packages = ["libgl1-mesa-glx", "libglu1-mesa"]

# Environment variables
[env_vars]
KMP_DUPLICATE_LIB_OK = "TRUE"
OMP_NUM_THREADS = "1"

# Dependent custom nodes to auto-install
[node_reqs]
ComfyUI_essentials = "cubiq/ComfyUI_essentials"

# Conda packages (via pixi)
[dependencies]
cgal = "*"

# PyPI packages
[pypi-dependencies]
trimesh = { version = "*", extras = ["easy"] }
numpy = "*"
```

---

## Process Isolation

For nodes with conflicting dependencies, use isolated execution:

```python
# In nodes/__init__.py
from pathlib import Path
from comfy_env import wrap_isolated_nodes

# Import your isolated nodes
from .cgal import NODE_CLASS_MAPPINGS as cgal_mappings

# Wrap them for isolated execution
NODE_CLASS_MAPPINGS = wrap_isolated_nodes(
    cgal_mappings,
    Path(__file__).parent / "cgal"  # Directory with comfy-env.toml
)
```

Each wrapped node runs in a subprocess with its own Python environment.

---

## CLI Commands

```bash
# Show detected environment
comfy-env info

# Install dependencies
comfy-env install

# Preview without installing
comfy-env install --dry-run

# Verify packages
comfy-env doctor

# Install system packages
comfy-env apt-install
```

---

## API Reference

### install()

Install dependencies from comfy-env.toml:

```python
from comfy_env import install

install()                    # Auto-detect config
install(dry_run=True)        # Preview only
install(config="path.toml")  # Explicit config
```

### setup_env()

Set up environment at ComfyUI startup:

```python
from comfy_env import setup_env

setup_env()  # Auto-detects node directory from caller
```

Sets library paths, environment variables, and injects site-packages.

### wrap_isolated_nodes()

Wrap nodes for subprocess isolation:

```python
from comfy_env import wrap_isolated_nodes

wrapped = wrap_isolated_nodes(NODE_CLASS_MAPPINGS, node_dir)
```

### Detection

```python
from comfy_env import (
    detect_cuda_version,      # Returns "12.8", "12.4", or None
    detect_gpu,               # Returns GPUInfo or None
    get_gpu_summary,          # Human-readable string
    RuntimeEnv,               # Combined runtime info
)

env = RuntimeEnv.detect()
print(env)  # Python 3.11, CUDA 12.8, PyTorch 2.8.0, GPU: RTX 4090
```

### Workers

Low-level process isolation:

```python
from comfy_env import MPWorker, SubprocessWorker

# Same Python version (multiprocessing)
worker = MPWorker()
result = worker.call(my_function, arg1, arg2)

# Different Python version (subprocess)
worker = SubprocessWorker(python="/path/to/python")
result = worker.call(my_function, arg1, arg2)
```

---

## Real Example

See [ComfyUI-GeometryPack](https://github.com/PozzettiAndrea/ComfyUI-GeometryPack) for a production example with:

- Multiple isolated environments (CGAL, Blender, GPU)
- Per-subdirectory comfy-env.toml
- Prestartup asset copying
- Different Python versions (3.11 for Blender API)

---

## Architecture

### Layers

```
comfy_env/
├── detection/     # Pure functions - CUDA, GPU, platform detection
├── config/        # Pure parsing - comfy-env.toml → typed config
├── environment/   # Side effects - cache, paths, setup
├── packages/      # Side effects - pixi, cuda-wheels, apt
├── isolation/     # Side effects - subprocess workers, node wrapping
└── install.py     # Orchestration
```

### Why Isolation?

ComfyUI nodes share a single Python environment. This breaks when:

1. **Dependency conflicts**: Node A needs `torch==2.4`, Node B needs `torch==2.8`
2. **Native library conflicts**: Two packages bundle incompatible libomp
3. **Python version requirements**: Blender API requires Python 3.11

Solution: Run each node group in its own subprocess with isolated dependencies.

### Why CUDA Wheels?

Installing packages like `nvdiffrast` normally requires:
- CUDA toolkit
- C++ compiler
- 30+ minutes of compilation

CUDA wheels from [cuda-wheels](https://pozzettiandrea.github.io/cuda-wheels/) are pre-built for common configurations:

| GPU | CUDA | PyTorch |
|-----|------|---------|
| Blackwell (sm_100+) | 12.8 | 2.8 |
| Ada/Hopper/Ampere | 12.8 | 2.8 |
| Turing | 12.8 | 2.8 |
| Pascal | 12.4 | 2.4 |

### How Environments Work

1. **Central cache**: Environments stored at `~/.comfy-env/envs/`
2. **Marker files**: `.comfy-env-marker.toml` links node → env
3. **Orphan cleanup**: Envs deleted when their node is removed
4. **Hash-based naming**: Config changes create new envs

---

## License

MIT
