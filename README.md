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
[env]
name = "my-node"
python = "3.10"
cuda = "auto"

[packages]
requirements = ["transformers>=4.56", "pillow"]
no_deps = ["nvdiffrast==0.4.0", "pytorch3d>=0.7.8"]

[sources]
wheel_sources = ["https://github.com/PozzettiAndrea/nvdiffrast-full-wheels/releases/download/"]
```

Then in your `__init__.py`:

```python
from comfy_env import install

# Install CUDA wheels into current environment
install()
```

### Process Isolation (Type 1 - Separate Venv)

For nodes that need completely separate dependencies:

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

# Verify installation
comfy-env doctor
```

## Configuration

### comfy-env.toml

```toml
[env]
name = "my-node"          # Unique name for caching
python = "3.10"           # Python version
cuda = "auto"             # "auto", "12.8", "12.4", or null

[packages]
requirements = [          # Regular pip packages
    "transformers>=4.56",
    "pillow",
]
no_deps = [               # CUDA packages (installed with --no-deps)
    "nvdiffrast==0.4.0",
    "pytorch3d>=0.7.8",
]

[sources]
wheel_sources = [         # GitHub releases with pre-built wheels
    "https://github.com/.../releases/download/",
]
index_urls = [            # Extra pip index URLs
    "https://pypi.org/simple/",
]

[worker]                  # For isolation mode
package = "worker"        # worker/__main__.py
```

### Template Variables

Wheel URLs support these template variables:

| Variable | Example | Description |
|----------|---------|-------------|
| `{cuda_version}` | `12.8` | Full CUDA version |
| `{cuda_short}` | `128` | CUDA without dot |
| `{torch_version}` | `2.8.0` | PyTorch version |
| `{torch_mm}` | `28` | PyTorch major.minor |
| `{py_version}` | `3.10` | Python version |
| `{py_short}` | `310` | Python without dot |
| `{platform}` | `linux_x86_64` | Platform tag |

## API Reference

### install()

```python
from comfy_env import install

# Auto-discover config
install()

# Explicit config
install(config="comfy-env.toml")

# Isolated mode (creates separate venv)
install(mode="isolated")

# Dry run
install(dry_run=True)
```

### WheelResolver

```python
from comfy_env import RuntimeEnv, WheelResolver

env = RuntimeEnv.detect()
resolver = WheelResolver()

url = resolver.resolve("nvdiffrast", "0.4.0", env)
print(url)  # https://github.com/.../nvdiffrast-0.4.0+cu128torch28-...whl
```

### Workers (for isolation)

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

## License

MIT - see LICENSE file.
