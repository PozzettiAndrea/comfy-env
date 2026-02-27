# Getting Started

## Install

```bash
pip install comfy-env
```

## Setup

A comfy-env node pack needs three integration files plus a root config. Here's the minimal setup:

### 1. Create `comfy-env-root.toml`

```toml
[apt]
packages = ["libgl1-mesa-glx"]

[node_reqs]
ComfyUI_essentials = "cubiq/ComfyUI_essentials"
```

This handles system packages and ComfyUI node dependencies. Python package dependencies go in `requirements.txt` as usual — ComfyUI installs those itself.

### 2. Create `install.py`

```python
from comfy_env import install
install()
```

Called by ComfyUI when the node pack is installed or updated. Installs system packages, clones node dependencies, and builds any isolated environments.

### 3. Create `prestartup_script.py`

```python
from comfy_env import setup_env
setup_env()
```

Called by ComfyUI before node loading. Sets up faulthandler, deduplicates libomp (macOS), and logs detected isolation environments.

### 4. Create `__init__.py`

```python
from comfy_env import register_nodes

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()
```

Discovers and registers all nodes. Subdirectories with a `comfy-env.toml` + built `_env_*` run in isolated subprocesses. Subdirectories without a config are imported normally.

## Adding an Isolated Subdirectory

To add a node that needs its own Python environment:

**1.** Create a subdirectory under your nodes folder with a `comfy-env.toml`:

```toml
python = "3.11"

[dependencies]
cgal = "*"

[pypi-dependencies]
trimesh = { version = "*", extras = ["easy"] }
```

**2.** Add your node's `__init__.py` in that subdirectory with standard ComfyUI node classes (`NODE_CLASS_MAPPINGS`, etc.).

**3.** Run `comfy-env install` or let ComfyUI trigger `install.py` — this builds the isolated environment.

You can also use `comfy-env init --isolated` to generate a starter `comfy-env.toml`.

## Directory Layout

```
ComfyUI-MyPack/
├── comfy-env-root.toml       # System packages + node deps
├── requirements.txt          # PyPI deps for main process (standard ComfyUI)
├── install.py                # comfy_env.install()
├── prestartup_script.py      # comfy_env.setup_env()
├── __init__.py               # comfy_env.register_nodes()
└── nodes/
    ├── main/                 # No comfy-env.toml → runs in main process
    │   └── __init__.py
    ├── cgal/                 # Isolated → subprocess with Python 3.11 + CGAL
    │   ├── comfy-env.toml
    │   ├── _env_a1b2c3/     # Link to cached build
    │   └── __init__.py
    └── gpu/                  # Isolated → subprocess with CUDA packages
        ├── comfy-env.toml
        ├── _env_d4e5f6/
        └── __init__.py
```

See [Config Reference](config-reference.md) for all available config options.
