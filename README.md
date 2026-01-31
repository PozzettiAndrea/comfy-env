# comfy-env

Environment management for ComfyUI custom nodes.

## Quick Start

```bash
pip install comfy-env
```

**1. Create `comfy-env-root.toml` in your node directory:**

```toml
[cuda]
packages = ["nvdiffrast", "pytorch3d"]

[apt]
packages = ["libgl1-mesa-glx"]

[node_reqs]
ComfyUI_essentials = "cubiq/ComfyUI_essentials"
```

PyPI deps go in `requirements.txt` (standard ComfyUI pattern).

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

---

## Two Config Files

| File | Purpose |
|------|---------|
| `comfy-env-root.toml` | Main node config (root level) |
| `comfy-env.toml` | Isolated subfolder config |

### comfy-env-root.toml (main node)

```toml
[cuda]
packages = ["nvdiffrast", "pytorch3d"]

[apt]
packages = ["libgl1-mesa-glx"]

[dependencies]
cgal = "*"

[env_vars]
KMP_DUPLICATE_LIB_OK = "TRUE"

[node_reqs]
ComfyUI_essentials = "cubiq/ComfyUI_essentials"
```

PyPI deps → `requirements.txt`

### comfy-env.toml (isolated folder)

```toml
python = "3.11"

[dependencies]
cgal = "*"

[pypi-dependencies]
trimesh = { version = "*", extras = ["easy"] }

[env_vars]
SOME_VAR = "value"
```

### What goes where?

| Section | Root | Isolated |
|---------|------|----------|
| `[cuda]` | ✓ | ✓ |
| `[apt]` | ✓ | ✓ |
| `[dependencies]` | ✓ | ✓ |
| `[env_vars]` | ✓ | ✓ |
| `[node_reqs]` | ✓ | ✗ |
| `python = "X.Y"` | ✗ | ✓ |
| `[pypi-dependencies]` | ✗ | ✓ |

---

## Process Isolation

For nodes with conflicting dependencies:

```python
# In nodes/__init__.py
from pathlib import Path
from comfy_env import wrap_isolated_nodes

from .cgal import NODE_CLASS_MAPPINGS as cgal_mappings

NODE_CLASS_MAPPINGS = wrap_isolated_nodes(
    cgal_mappings,
    Path(__file__).parent / "cgal"  # Has comfy-env.toml
)
```

Each wrapped node runs in a subprocess with its own Python environment.

---

## CLI

```bash
comfy-env init              # Create comfy-env-root.toml
comfy-env init --isolated   # Create comfy-env.toml (for subfolders)
comfy-env install           # Install dependencies
comfy-env install --dry-run # Preview
comfy-env info              # Show runtime info
comfy-env doctor            # Verify packages
comfy-env apt-install       # Install system packages
```

---

## API

### install()

```python
from comfy_env import install
install()
```

### setup_env()

```python
from comfy_env import setup_env
setup_env()  # Call in prestartup_script.py
```

### wrap_isolated_nodes()

```python
from comfy_env import wrap_isolated_nodes
wrapped = wrap_isolated_nodes(NODE_CLASS_MAPPINGS, node_dir)
```

### Detection

```python
from comfy_env import RuntimeEnv, detect_cuda_version, detect_gpu

env = RuntimeEnv.detect()
print(env)  # Python 3.11, CUDA 12.8, PyTorch 2.8.0, GPU: RTX 4090
```

---

## Example

See [ComfyUI-GeometryPack](https://github.com/PozzettiAndrea/ComfyUI-GeometryPack):

- Multiple isolated environments (CGAL, Blender, GPU)
- Per-subdirectory `comfy-env.toml`
- Different Python versions

---

## Why?

**Why isolation?** ComfyUI nodes share one Python. Conflicts happen when:
- Node A needs torch 2.4, Node B needs torch 2.8
- Two packages bundle incompatible libomp
- Blender API requires Python 3.11

**Why CUDA wheels?** Installing nvdiffrast normally needs CUDA toolkit + C++ compiler + 30 min compilation. [cuda-wheels](https://pozzettiandrea.github.io/cuda-wheels/) provides pre-built wheels.

**How envs work:**
- Central cache: `~/.comfy-env/envs/`
- Marker files link nodes → cached envs
- Config hash in name → changes create new envs

---

## License

MIT
