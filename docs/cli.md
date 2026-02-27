# CLI Reference

```bash
comfy-env [command] [options]
```

## Commands

### `init`

Create a starter config file in the current directory.

```bash
comfy-env init              # Creates comfy-env-root.toml
comfy-env init --isolated   # Creates comfy-env.toml (for subdirectories)
```

| Flag | Description |
|------|-------------|
| `--force`, `-f` | Overwrite existing config |
| `--isolated` | Create `comfy-env.toml` instead of `comfy-env-root.toml` |

### `install`

Install all dependencies (system packages, node deps, isolated environments).

```bash
comfy-env install
comfy-env install --dry-run
comfy-env install --config path/to/config.toml --dir path/to/node
```

| Flag | Description |
|------|-------------|
| `--config`, `-c` | Path to config file |
| `--dir`, `-d` | Node directory (default: current directory) |
| `--dry-run` | Preview what would be installed |

### `generate`

Generate a `pixi.toml` from a `comfy-env.toml` config. Useful for debugging or manual pixi usage.

```bash
comfy-env generate path/to/comfy-env.toml
```

| Flag | Description |
|------|-------------|
| `--force`, `-f` | Overwrite existing `pixi.toml` |

### `info`

Show detected runtime environment (OS, Python, CUDA, PyTorch, GPU).

```bash
comfy-env info
comfy-env info --json
```

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON |

### `doctor`

Verify that packages from a config are correctly installed.

```bash
comfy-env doctor
comfy-env doctor --package numpy
comfy-env doctor --config path/to/config.toml
```

| Flag | Description |
|------|-------------|
| `--package`, `-p` | Check a specific package |
| `--config`, `-c` | Path to config file |

### `apt-install`

Install system packages from config (Linux only).

```bash
comfy-env apt-install
comfy-env apt-install --dry-run
```

| Flag | Description |
|------|-------------|
| `--config`, `-c` | Path to config file |
| `--dry-run` | Preview what would be installed |

Requires root or sudo. Reads `[apt]` packages from `comfy-env-root.toml` (or `comfy-env.toml` if root config doesn't exist).

### `cleanup`

Deprecated. Environments are stored in the central build cache (`~/.ce` or `C:/ce`). To remove an environment, delete the corresponding `_env_*` directory in the node folder and optionally the matching directory in the cache.
