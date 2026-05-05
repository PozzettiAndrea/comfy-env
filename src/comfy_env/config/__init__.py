"""comfy-env configuration. One file, no bullshit."""

from pathlib import Path

import tomli

ROOT_CONFIG_FILE_NAME = "comfy-env-root.toml"
CONFIG_FILE_NAME = "comfy-env.toml"
DEFAULT_HEALTH_CHECK_TIMEOUT = 5.0


class ComfyEnvConfig(dict):
    """Config is just a dict you can also access with dot notation."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @property
    def has_cuda(self):
        return bool(self.get("cuda_packages"))

    @property
    def has_dependencies(self):
        return bool(
            self.get("cuda_packages") or self.get("node_reqs")
            or self.get("pixi_passthrough", {}).get("dependencies")
            or self.get("pixi_passthrough", {}).get("pypi-dependencies")
        )


# Backwards compat alias
NodeDependency = dict


def load_config(path):
    """Load a comfy-env TOML file. Returns a ComfyEnvConfig."""
    with open(path, "rb") as f:
        return parse_config(tomli.load(f))


def discover_config(node_dir, root=True):
    """Find config in a directory. Returns parsed dict or None."""
    node_dir = Path(node_dir)
    if root:
        p = node_dir / ROOT_CONFIG_FILE_NAME
        if p.exists():
            return load_config(p)
    p = node_dir / CONFIG_FILE_NAME
    return load_config(p) if p.exists() else None


def parse_config(data):
    """Parse raw TOML dict into a normalized config dict.

    Returns:
        {
            "python": str | None,
            "cuda_packages": [str],
            "env_vars": {str: str},
            "node_reqs": [{"name": str, "github": str|None, "tag": str|None, ...}],
            "options": {"health_check_timeout": float},
            "settings": dict,
            "pixi_passthrough": dict,  # everything else goes straight to pixi.toml
        }
    """
    data = dict(data)  # shallow copy

    python = data.pop("python", None)
    if python is not None:
        python = str(python)

    cuda = data.pop("cuda", {})
    cuda_packages = cuda.get("packages", [])
    if not isinstance(cuda_packages, list):
        cuda_packages = [cuda_packages] if cuda_packages else []

    data.pop("apt", None)
    data.pop("brew", None)

    env_vars = {str(k): str(v) for k, v in data.pop("env_vars", {}).items()}
    node_reqs = _parse_node_reqs(data.pop("node_reqs", {}))
    options = data.pop("options", {})
    settings = data.pop("settings", {})

    return ComfyEnvConfig(
        python=python,
        cuda_packages=cuda_packages,
        env_vars=env_vars,
        node_reqs=node_reqs,
        options={"health_check_timeout": float(options.get("health_check_timeout", DEFAULT_HEALTH_CHECK_TIMEOUT))},
        settings=settings,
        pixi_passthrough=data,
    )


def _parse_node_reqs(data):
    reqs = []
    for name, value in data.items():
        if isinstance(value, str):
            reqs.append({"name": name, "github": value})
        else:
            reqs.append({
                "name": name,
                "github": value.get("github") or value.get("repo"),
                "tag": value.get("tag"),
                "branch": value.get("branch"),
                "commit": value.get("commit"),
                "registry": value.get("registry"),
                "version": value.get("version"),
            })
    return reqs
