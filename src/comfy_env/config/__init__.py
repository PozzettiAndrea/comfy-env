import re
import sys
from pathlib import Path

import tomli

#Two config types:
# one that sits at root (ComfyUI-Geometrypack/comfy-env-root.toml)
ROOT_CONFIG_FILE_NAME = "comfy-env-root.toml"
# one that sits at node folder (ComfyUI-Sharp/nodes/comfy-env.toml)
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
            self.get("cuda_packages") or self.get("node_packs")
            or self.get("pixi_passthrough", {}).get("dependencies")
            or self.get("pixi_passthrough", {}).get("pypi-dependencies")
        )

# Role schemas. The root file carries pack-level declarations only; the env
# file carries an env definition and must NOT carry the root-only sections.
# Anything role-inappropriate -- dead legacy keys, typos, misplaced sections
# -- is rejected at parse time rather than silently ignored (that's how a
# no-op [env_vars] shipped in the flagship pack for months).
ROOT_ALLOWED_SECTIONS = {"node_packs", "types"}
# Deliberately the SAME object, not a second literal: "allowed at root" and
# "root-only" are different questions that happen to have the same answer,
# and two literals drift the moment someone adds a section to one.
ROOT_ONLY_SECTIONS = ROOT_ALLOWED_SECTIONS

# Comfy-env-owned sections of the ENV file and their known keys: the one
# place pixi cannot validate for us, so unrecognized keys warn (a typo'd
# `pakages` otherwise vanishes without a trace). Everything outside these
# sections is pixi's language and is validated by pixi (ADR-0013).
_OWNED_SECTION_KEYS = {
    "cuda": {"packages"},
    "options": {"health_check_timeout"},
}


def load_config(path):
    """Load a comfy-env TOML file. Returns a ComfyEnvConfig.

    The filename determines the role: comfy-env-root.toml is validated
    against the closed root schema; comfy-env.toml rejects root-only
    sections (they were parsed-and-ignored for months -- no more).
    """
    path = Path(path)
    with open(path, "rb") as f:
        data = tomli.load(f)
    if path.name == ROOT_CONFIG_FILE_NAME:
        unknown = sorted(set(data) - ROOT_ALLOWED_SECTIONS)
        if unknown:
            raise ValueError(
                f"{path}: unsupported section(s) "
                f"{', '.join('[' + s + ']' for s in unknown)} -- "
                f"{ROOT_CONFIG_FILE_NAME} carries [node_packs] and [types] "
                f"only ([settings] was removed in 0.4.25 -- the surviving "
                f"settings are machine-global env vars). Env definitions "
                f"(dependencies, cuda, env_vars, ...) go in a subdirectory "
                f"{CONFIG_FILE_NAME}.")
    elif path.name == CONFIG_FILE_NAME:
        if "serializers" in data:
            raise ValueError(
                f"{path}: [serializers] was replaced by [types] in "
                f"{ROOT_CONFIG_FILE_NAME} at the pack root (comfy-env "
                f"0.4.16, ADR-0015). Declare each socket as \"builtin\" or "
                f"\"custom\" and put custom serializers in "
                f"<pack>/serialization.py.")
        misplaced = sorted(ROOT_ONLY_SECTIONS & set(data))
        if misplaced:
            raise ValueError(
                f"{path}: section(s) "
                f"{', '.join('[' + s + ']' for s in misplaced)} belong in "
                f"{ROOT_CONFIG_FILE_NAME} at the pack root -- in an env file "
                f"they have never had any effect.")
    return parse_config(data)


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
    """Parse a raw TOML dict into the normalized config dict built below.

    Everything comfy-env does not claim lands in ``pixi_passthrough`` and goes
    to pixi verbatim.
    """
    data = dict(data)  # shallow copy

    # Typo guard for comfy-env-owned sections -- pixi validates everything
    # else, but these are ours (ADR-0013).
    owned = dict(_OWNED_SECTION_KEYS)
    for section, known in owned.items():
        table = data.get(section)
        if isinstance(table, dict):
            for key in table:
                if key not in known:
                    print(
                        f"[comfy-env] WARNING: unrecognized key '{key}' in "
                        f"[{section}] (known: {', '.join(sorted(known))})",
                        file=sys.stderr, flush=True)

    python = data.pop("python", None)
    if python is not None:
        # TOML gotcha: unquoted `python = 3.10` is a FLOAT and becomes 3.1
        # (3.11/3.12 unquoted survive by luck, which is what makes this a
        # landmine). Require a string.
        if not isinstance(python, str):
            raise ValueError(
                f'python = {python} is not a string (unquoted TOML numbers '
                f'lose trailing zeros: 3.10 becomes 3.1). Quote it: '
                f'python = "{python}"')
        python = str(python)
        # Floor: ComfyUI itself is requires-python >= 3.10, and the worker
        # program is only kept parseable down to 3.10 (ADR-0006). A lower pin
        # would fail at worker startup, long after a multi-GB env build.
        m = re.search(r"3\.(\d+)", python)
        if m and int(m.group(1)) < 10:
            raise ValueError(
                f'python = "{python}" pins below 3.10. comfy-env supports '
                f'Python >= 3.10 only (ComfyUI itself requires >= 3.10, and '
                f'the worker program is 3.10+).')

    cuda = data.pop("cuda", {})
    cuda_packages = cuda.get("packages", [])
    if not isinstance(cuda_packages, list):
        cuda_packages = [cuda_packages] if cuda_packages else []

    env_vars = {str(k): str(v) for k, v in data.pop("env_vars", {}).items()}
    node_packs = _parse_node_packs(data.pop("node_packs", {}))
    options = data.pop("options", {})

    # [types] (ADR-0015): SOCKET = "builtin" | "custom". Closed vocabulary
    # -- anything else is a typo'd contract and fails at parse time.
    types = data.pop("types", {})
    for socket, mode in types.items():
        if mode not in ("builtin", "custom"):
            raise ValueError(
                f'[types] {socket} = "{mode}" is not valid -- use "builtin" '
                f'(automatic transport) or "custom" (serialize/deserialize '
                f'functions in <pack>/serialization.py).')

    return ComfyEnvConfig(
        python=python,
        cuda_packages=cuda_packages,
        env_vars=env_vars,
        node_packs=node_packs,
        options={"health_check_timeout": float(options.get("health_check_timeout", DEFAULT_HEALTH_CHECK_TIMEOUT))},
        types={str(k): str(v) for k, v in types.items()},
        pixi_passthrough=data,
    )


def _parse_node_packs(data):
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
