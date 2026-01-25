# comfy-env

Environment management for ComfyUI custom nodes. Provides:

1. **CUDA Wheel Resolution** - Install pre-built CUDA wheels (nvdiffrast, pytorch3d) without compilation
2. **Process Isolation** - Run nodes in separate Python environments with their own dependencies

## Why?

ComfyUI custom nodes face two challenges:

**Type 1: Dependency Conflicts**
- Node A needs `torch==2.1.0` with CUDA 11.8
- Node B needs `torch==2.8.0` with CUDA 12.8

**Type 2: CUDA Package Installation**
- Users don't have compilers installed
- Building from source takes forever
- pip install fails with cryptic errors

This package solves both problems.

## Installation

```bash
pip install comfy-env
```

Requires [uv](https://github.com/astral-sh/uv) for fast environment creation:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

### In-Place Installation (Type 2 - CUDA Wheels)

Create a `comfy-env.toml` in your node directory:

```toml
[cuda]
nvdiffrast = "0.4.0"
pytorch3d = "0.7.9"

[packages]
requirements = ["transformers>=4.56", "pillow"]
```

Then in your `__init__.py`:

```python
from comfy_env import install

# Install CUDA wheels into current environment
install()
```

### Process Isolation (Type 1 - Separate Environment)

For nodes that need completely separate dependencies (different Python version, conda packages, conflicting libraries).

#### Recommended: Pack-Wide Isolation

For node packs where ALL nodes run in the same isolated environment:

**Step 1: Configure comfy-env.toml**

```toml
[mypack]
python = "3.11"
isolated = true          # All nodes run in this env

[mypack.conda]
packages = ["cgal"]      # Conda packages (uses pixi)

[mypack.packages]
requirements = ["trimesh[easy]>=4.0", "bpy>=4.2"]
```

**Step 2: Enable in __init__.py**

```python
from comfy_env import setup_isolated_imports, enable_isolation

# Setup import stubs BEFORE importing nodes
setup_isolated_imports(__file__)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Enable isolation for all nodes
enable_isolation(NODE_CLASS_MAPPINGS)
```

**That's it!** All nodes run in an isolated Python 3.11 environment with their own dependencies.

#### Alternative: Per-Node Isolation

For cases where different nodes need different environments:

```python
from comfy_env import isolated

@isolated(env="my-node")
class MyNode:
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE",)

    def process(self, image):
        # Runs in isolated subprocess with its own venv
        import conflicting_package
        return (result,)
```

## CLI

```bash
# Show detected environment
comfy-env info

# Install from config
comfy-env install

# Dry run (show what would be installed)
comfy-env install --dry-run

# Resolve wheel URLs without installing
comfy-env resolve nvdiffrast==0.4.0

# List all packages in the built-in registry
comfy-env list-packages

# Verify installation
comfy-env doctor
```

## Configuration

### Simple Format (comfy-env.toml)

```toml
# CUDA packages (uses built-in registry)
[cuda]
nvdiffrast = "0.4.0"
pytorch3d = "0.7.9"
torch-scatter = "2.1.2"

# Regular pip packages
[packages]
requirements = ["transformers>=4.56", "pillow"]
```

### Full Format

```toml
[system]
linux = ["libgl1", "libopengl0"]  # apt packages

[local.cuda]
nvdiffrast = "0.4.0"

[local.packages]
requirements = ["pillow", "numpy"]

# For isolated environments (creates separate venv)
[myenv]
python = "3.10"
cuda = "12.8"

[myenv.cuda]
torch-scatter = "2.1.2"

[myenv.packages]
requirements = ["transformers>=4.56"]

# Custom wheel templates (override built-in registry)
[wheel_sources]
my-custom-pkg = "https://my-server.com/my-pkg-{version}+cu{cuda_short}-{py_tag}-{platform}.whl"
```

## Writing Wheel Templates

### Template Variables

| Variable | Example | Description |
|----------|---------|-------------|
| `{version}` | `0.4.0` | Package version |
| `{cuda_version}` | `12.8` | Full CUDA version |
| `{cuda_short}` | `128` | CUDA without dot |
| `{cuda_major}` | `12` | CUDA major only |
| `{torch_version}` | `2.8.0` | Full PyTorch version |
| `{torch_mm}` | `28` | PyTorch major.minor no dot |
| `{torch_dotted_mm}` | `2.8` | PyTorch major.minor with dot |
| `{py_version}` | `3.10` | Python version |
| `{py_short}` | `310` | Python without dot |
| `{py_tag}` | `cp310` | Python wheel tag |
| `{platform}` | `linux_x86_64` | Platform tag |

### Common Wheel URL Patterns

**Pattern 1: Simple CUDA + Python**
```
https://example.com/{package}-{version}+cu{cuda_short}-{py_tag}-{py_tag}-{platform}.whl
```

**Pattern 2: CUDA + PyTorch**
```
https://example.com/{package}-{version}+cu{cuda_short}torch{torch_mm}-{py_tag}-{py_tag}-{platform}.whl
```

**Pattern 3: GitHub Releases**
```
https://github.com/org/repo/releases/download/v{version}/{package}-{version}+cu{cuda_short}-{py_tag}-{platform}.whl
```

### How to Find the Right Template

1. Download a wheel manually from the source
2. Look at the filename pattern: `nvdiffrast-0.4.0+cu128torch28-cp310-cp310-linux_x86_64.whl`
3. Replace values with variables: `nvdiffrast-{version}+cu{cuda_short}torch{torch_mm}-{py_tag}-{py_tag}-{platform}.whl`
4. Prepend the base URL

### Testing Your Template

```bash
comfy-env resolve my-package==1.0.0
```

This shows the resolved URL without installing.

### Adding Custom Wheel Sources

If a package isn't in the built-in registry, add it to your `comfy-env.toml`:

```toml
[cuda]
my-custom-pkg = "1.0.0"

[wheel_sources]
my-custom-pkg = "https://my-server.com/my-custom-pkg-{version}+cu{cuda_short}-{py_tag}-{platform}.whl"
```

Resolution order:
1. User's `[wheel_sources]` in comfy-env.toml (highest priority)
2. Built-in `wheel_sources.yml` registry
3. Error if not found

## API Reference

### install()

```python
from comfy_env import install

# Auto-discover config
install()

# Explicit config
install(config="comfy-env.toml")

# Dry run
install(dry_run=True)
```

### RuntimeEnv

```python
from comfy_env import RuntimeEnv

env = RuntimeEnv.detect()
print(env)
# Python 3.10, CUDA 12.8, PyTorch 2.8.0, GPU: NVIDIA GeForce RTX 4090

# Get template variables
vars_dict = env.as_dict()
# {'cuda_version': '12.8', 'cuda_short': '128', 'torch_mm': '28', ...}
```

### enable_isolation()

```python
from comfy_env import enable_isolation

enable_isolation(NODE_CLASS_MAPPINGS)
```

Wraps all node classes so their FUNCTION methods run in the isolated environment specified in comfy-env.toml. Requires `isolated = true` in the environment config.

### setup_isolated_imports()

```python
from comfy_env import setup_isolated_imports

setup_isolated_imports(__file__)
```

Sets up import stubs for packages that exist only in the isolated pixi environment. Call this BEFORE importing your nodes module. Packages available in both host and isolated environment are not stubbed.

### Workers (for custom isolation)

```python
from comfy_env import TorchMPWorker

# Same-venv isolation (zero-copy tensors)
worker = TorchMPWorker()
result = worker.call(my_function, image=tensor)
```

## GPU Detection

```python
from comfy_env import detect_cuda_version, get_gpu_summary

cuda = detect_cuda_version()  # "12.8", "12.4", or None
print(get_gpu_summary())
# GPU 0: NVIDIA GeForce RTX 5090 (sm_120) [Blackwell - CUDA 12.8]
```

## Built-in Package Registry

Run `comfy-env list-packages` to see all packages in the built-in registry.

The registry includes:
- PyTorch Geometric packages (torch-scatter, torch-cluster, torch-sparse)
- NVIDIA packages (nvdiffrast, pytorch3d, gsplat)
- Flash Attention (flash-attn)
- And more

## License

MIT - see LICENSE file.
