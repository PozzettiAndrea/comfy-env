"""Generate the workspace pixi.toml from ComfyUI requirements + per-node configs.

Workspace model:
- One pixi workspace per ComfyUI install at `<comfyui_dir>/.ce/pixi.toml`.
- `[feature.comfyui.pypi-dependencies]` is parsed from `<comfyui_dir>/requirements.txt`
  with torch/torchvision/torchaudio pointed at the resolved CUDA/CPU index.
- `[feature.py<XY>.dependencies]` per python version actually requested by any node.
- `[feature.<env_name>.*]` per node config.
- `[environments]` composes (pyXY + comfyui + node) per env, all `no-default-feature = true`.
"""

import copy
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import ComfyEnvConfig
from ..detection import get_pixi_platform


_TORCH_PKGS = {"torch", "torchvision", "torchaudio"}


def _require_tomli_w():
    try:
        import tomli_w
        return tomli_w
    except ImportError:
        raise ImportError("tomli-w required: pip install tomli-w")


# ---------------------------------------------------------------------------
# requirements.txt parsing
# ---------------------------------------------------------------------------

_REQ_LINE_RE = re.compile(
    r"""
    ^
    (?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)              # package name
    (?: \[ (?P<extras>[^\]]+) \] )?                   # optional [extras]
    (?P<spec>                                          # version spec(s)
        (?:\s*(?:==|>=|<=|>|<|~=|!=)\s*[^,\s;#]+)*
    )?
    \s*$
    """,
    re.VERBOSE,
)


def _normalize_spec(spec: str) -> str:
    """Compact a comma-joined version spec by stripping spaces. ' >= 1.2 ' -> '>=1.2'."""
    return spec.strip().replace(" ", "")


def parse_requirement_line(line: str) -> Optional[Tuple[str, Any]]:
    """Parse one pip requirement line into a (name, pixi-spec) pair.

    Returns None for blank lines, comments, options (`-r`, `--index-url`, etc.),
    and lines that don't match the simple pattern (e.g. URL-only entries we don't
    yet translate).

    Examples:
        "torch==2.8.0"             -> ("torch", "==2.8.0")
        "numpy>=1.25.0"            -> ("numpy", ">=1.25.0")
        "Pillow"                   -> ("Pillow", "*")
        "trimesh[easy]>=4.0.0"     -> ("trimesh", {"version": ">=4.0.0", "extras": ["easy"]})
        "pydantic~=2.0"            -> ("pydantic", "~=2.0")
        "# comment"                -> None
    """
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    # Strip trailing inline comment (# is only a comment if preceded by whitespace
    # or at start; pip requirements use ; for env markers but those are also dropped here)
    text = text.split(";", 1)[0].strip()
    if not text:
        return None
    if " #" in text:
        text = text.split(" #", 1)[0].strip()
    if text.startswith("-"):
        return None  # options handled by caller (-r, --index-url, etc.)
    if "://" in text or text.startswith("git+"):
        # URL/VCS entries: skip -- pixi has a separate syntax for these. Caller
        # can handle them out-of-band if needed.
        return None
    m = _REQ_LINE_RE.match(text)
    if not m:
        return None
    name = m.group("name")
    extras_str = m.group("extras") or ""
    extras = [e.strip() for e in extras_str.split(",") if e.strip()]
    spec = _normalize_spec(m.group("spec") or "")
    if not spec:
        spec = "*"
    if extras:
        return name, {"version": spec, "extras": extras}
    return name, spec


def parse_comfyui_requirements(
    comfyui_dir: Path,
    torch_index: Optional[str],
    log: Callable[[str], None] = print,
    _seen: Optional[set] = None,
) -> Dict[str, Any]:
    """Read `<comfyui_dir>/requirements.txt` and produce a pixi pypi-dependencies dict.

    torch/torchvision/torchaudio entries get `index = torch_index` attached so uv
    fetches them from the pytorch wheel index (CPU or CUDA, decided once for the
    workspace). `-r other.txt` lines are followed recursively.
    """
    _seen = _seen if _seen is not None else set()
    req_file = Path(comfyui_dir) / "requirements.txt"
    return _parse_requirements_file(req_file, torch_index, log, _seen)


def _parse_requirements_file(
    req_file: Path,
    torch_index: Optional[str],
    log: Callable[[str], None],
    seen: set,
) -> Dict[str, Any]:
    if not req_file.exists():
        log(f"[comfy-env] WARNING: {req_file} not found -- comfyui feature will be empty")
        return {}
    real = req_file.resolve()
    if real in seen:
        return {}
    seen.add(real)

    out: Dict[str, Any] = {}
    for raw in req_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r ") or line.startswith("--requirement "):
            sub = line.split(maxsplit=1)[1].strip()
            sub_path = (req_file.parent / sub).resolve()
            out.update(_parse_requirements_file(sub_path, torch_index, log, seen))
            continue
        if line.startswith("--index-url") or line.startswith("--extra-index-url") or line.startswith("-i "):
            log(f"[comfy-env] Note: ignoring index directive in {req_file.name}: {line}")
            continue
        if line.startswith("-"):
            continue
        parsed = parse_requirement_line(line)
        if parsed is None:
            log(f"[comfy-env] WARNING: skipping unparseable requirement: {line!r}")
            continue
        name, spec = parsed
        # Attach torch index for torch family
        if name.lower() in _TORCH_PKGS and torch_index:
            if isinstance(spec, dict):
                spec = {**spec, "index": torch_index}
            else:
                spec = {"version": spec, "index": torch_index}
        out[name] = spec
    return out


# ---------------------------------------------------------------------------
# Workspace builder
# ---------------------------------------------------------------------------

def _python_feature_name(version: str) -> str:
    """3.11 -> py311, 3.13 -> py313."""
    parts = version.split(".")
    return "py" + parts[0] + parts[1]


def _build_python_feature(version: str) -> Dict[str, Any]:
    return {
        "dependencies": {
            "python": f"{version}.*",
            "pip": "*",
            "setuptools": ">=75.0,<82",
        }
    }


def _validate_node_config(name: str, cfg: ComfyEnvConfig) -> None:
    """Reject node configs that try to redefine workspace-global torch."""
    bad = [p for p in cfg.cuda_packages if p in _TORCH_PKGS]
    if bad:
        raise ValueError(
            f"[{name}] comfy-env.toml has {bad} under [cuda] packages. "
            "Plain torch/torchvision/torchaudio are now provided by the "
            "workspace's `comfyui` feature (parsed from <ComfyUI>/requirements.txt). "
            "Remove them from [cuda] packages -- keep only CUDA-only wheels there "
            "(cumesh, flash-attn, cc_torch, nvdiffrast, etc.)."
        )


def _strip_torch_family(
    table: Dict[str, Any],
    name: str,
    where: str,
    log: Callable[[str], None],
) -> None:
    """Remove plain torch/torchvision/torchaudio entries from a deps table in place.

    Torch family is workspace-global (provided by the `comfyui` feature). Per-node
    declarations are silently stripped with a one-line note.
    """
    for k in list(table.keys()):
        if k.lower() in _TORCH_PKGS:
            del table[k]
            log(f"[comfy-env] {name}: ignoring `{k}` in {where} (provided by workspace `comfyui` feature)")


def _build_node_feature(
    cfg: ComfyEnvConfig, name: str, log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Emit a pixi `[feature.<name>.*]` block from a node's ComfyEnvConfig.

    Only carries node-specific deps. Python pin lives in the pyXY feature; torch
    lives in the comfyui feature. No platform gates -- pypi index resolution
    handles wheel selection. Plain torch/torchvision/torchaudio entries are
    stripped (workspace-global from the `comfyui` feature).
    """
    feat: Dict[str, Any] = {}

    deps = copy.deepcopy(cfg.pixi_passthrough.get("dependencies", {}))
    if deps:
        _strip_torch_family(deps, name, "[dependencies]", log)
        if deps:
            feat["dependencies"] = deps

    pypi = copy.deepcopy(cfg.pixi_passthrough.get("pypi-dependencies", {}))
    if pypi:
        _strip_torch_family(pypi, name, "[pypi-dependencies]", log)
        if pypi:
            feat["pypi-dependencies"] = pypi

    # Per-target sections (only the current platform's), with torch family also stripped
    targets = cfg.pixi_passthrough.get("target", {})
    current = get_pixi_platform()
    if current in targets:
        cur_target = copy.deepcopy(targets[current])
        for tbl in ("dependencies", "pypi-dependencies"):
            if tbl in cur_target:
                _strip_torch_family(
                    cur_target[tbl], name,
                    f"[target.{current}.{tbl}]", log,
                )
                if not cur_target[tbl]:
                    del cur_target[tbl]
        if cur_target:
            feat.setdefault("target", {})[current] = cur_target

    return feat


def _pin_torch_family(
    pypi: Dict[str, Any],
    torch_pin: Optional[str],
    log: Callable[[str], None],
) -> None:
    """Pin torch and derive matching torchvision/torchaudio pins from TORCH_FAMILY_COMPAT.

    `torch_pin` is a PEP 440 spec like `"==2.11.0"` (tier-1) or `"==2.8.*"` (tier-2
    fallback). The torch INDEX (cu{NN} / cpu) was attached upstream by
    parse_comfyui_requirements; here we just clamp versions so per-node envs
    hardlink-share an identical torch family with the comfyui template env.

    torchvision and torchaudio have their own version numbering (torch 2.11 pairs
    with torchvision 0.26 and torchaudio 2.11), so the siblings come from the
    compat table rather than reusing torch's literal version string. If torch's
    minor isn't in the table, only torch is pinned and a warning is logged.

    No-op when `torch_pin` is None.
    """
    if not torch_pin:
        return
    from .cuda_wheels import derive_family_pins
    family = derive_family_pins(torch_pin)
    if family is None:
        log(
            f"[comfy-env] WARNING: torch_pin {torch_pin} not in TORCH_FAMILY_COMPAT; "
            f"pinning torch only, leaving torchvision/torchaudio unpinned"
        )
        pin_map = {"torch": torch_pin}
    else:
        vision_pin, audio_pin = family
        pin_map = {"torch": torch_pin, "torchvision": vision_pin, "torchaudio": audio_pin}

    for k in list(pypi.keys()):
        new_spec = pin_map.get(k.lower())
        if new_spec is None:
            continue
        spec = pypi[k]
        if isinstance(spec, dict):
            pypi[k] = {**spec, "version": new_spec}
        else:
            pypi[k] = new_spec
        log(f"[comfy-env] comfyui: pinning {k} {new_spec}")


def build_workspace_toml(
    comfyui_dir: Path,
    torch_index: Optional[str],
    cuda_major: Optional[str],
    node_configs: List[Tuple[str, ComfyEnvConfig]],  # (env_name, cfg) pairs
    bootstrap_python: Optional[str] = None,
    torch_pin: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Assemble the full workspace pixi.toml as a dict.

    Args:
        comfyui_dir: ComfyUI install root (used to find requirements.txt).
        torch_index: PyPI index URL for torch wheels (e.g. ".../whl/cpu" or ".../whl/cu124").
        cuda_major: Major CUDA version for `[system-requirements]` (e.g. "12"). None on CPU.
        node_configs: List of (env_name, ComfyEnvConfig) -- one entry per environment to create.
        bootstrap_python: Major.minor of the install bootstrap interpreter (e.g. "3.10").
            Defaults to `sys.version_info` on the running interpreter.
        torch_pin: PEP 440 version spec to pin torch/torchvision/torchaudio
            (e.g. `"==2.11.0"` for tier-1 bootstrap match, `"==2.8.*"` for tier-2
            fallback). When None, torch stays as `*` from ComfyUI's requirements.txt.
    """
    host_py = bootstrap_python or f"{sys.version_info.major}.{sys.version_info.minor}"
    current_platform = get_pixi_platform()

    # Validate first so we fail fast on bad node configs
    for name, cfg in node_configs:
        _validate_node_config(name, cfg)

    out: Dict[str, Any] = {
        "workspace": {
            "name": "comfy-env",
            "version": "0.1.0",
            "channels": ["conda-forge"],
            "platforms": [current_platform],
        }
    }

    # comfyui baseline feature
    comfyui_pypi = parse_comfyui_requirements(comfyui_dir, torch_index, log)
    _pin_torch_family(comfyui_pypi, torch_pin, log)
    feature_comfyui: Dict[str, Any] = {}
    if comfyui_pypi:
        feature_comfyui["pypi-dependencies"] = comfyui_pypi
    if cuda_major:
        feature_comfyui["system-requirements"] = {"cuda": cuda_major}
    if sys.platform.startswith("linux"):
        feature_comfyui.setdefault("system-requirements", {}).setdefault(
            "libc", {"family": "glibc", "version": "2.35"}
        )
    out["feature"] = {"comfyui": feature_comfyui}

    # Per-python features
    py_versions: Dict[str, str] = {host_py: _python_feature_name(host_py)}
    for _, cfg in node_configs:
        v = cfg.python or host_py
        py_versions.setdefault(v, _python_feature_name(v))
    for v, fname in py_versions.items():
        out["feature"][fname] = _build_python_feature(v)

    # Per-node features
    for env_name, cfg in node_configs:
        feat = _build_node_feature(cfg, env_name, log)
        if feat:
            out["feature"][env_name] = feat
        # if a node has zero pixi-passthrough deps, still create an (empty) feature
        # so the env composes correctly.
        else:
            out["feature"][env_name] = {}

    # Environments table.
    # The `comfyui` template env exists for every workspace -- it pins what the
    # main ComfyUI process is running and gives the cuda-wheel picker a canonical
    # place to read torch's resolved version from. Per-node envs share the same
    # solve-group so torch is hardlinked across envs from a single content cache.
    host_py_feature = py_versions[host_py]
    environments: Dict[str, Any] = {
        "comfyui": {
            "features": [host_py_feature, "comfyui"],
            "no-default-feature": True,
            "solve-group": host_py_feature,
        }
    }
    for env_name, cfg in node_configs:
        v = cfg.python or host_py
        py_feature = py_versions[v]
        environments[env_name] = {
            "features": [py_feature, "comfyui", env_name],
            "no-default-feature": True,
            "solve-group": py_feature,
        }
    out["environments"] = environments

    return out


# ---------------------------------------------------------------------------
# Top-level write entry point
# ---------------------------------------------------------------------------

def write_workspace_pixi_toml(
    workspace_dir: Path,
    comfyui_dir: Path,
    torch_index: Optional[str],
    cuda_major: Optional[str],
    node_configs: List[Tuple[str, ComfyEnvConfig]],
    bootstrap_python: Optional[str] = None,
    torch_pin: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Generate `<workspace_dir>/pixi.toml` from the parts above. Returns the file path."""
    tomli_w = _require_tomli_w()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    pixi_toml = workspace_dir / "pixi.toml"
    data = build_workspace_toml(
        comfyui_dir, torch_index, cuda_major, node_configs,
        bootstrap_python=bootstrap_python,
        torch_pin=torch_pin,
        log=log,
    )
    with open(pixi_toml, "wb") as f:
        tomli_w.dump(data, f)
    log(f"Generated {pixi_toml}")
    return pixi_toml


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result
