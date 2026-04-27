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
    auto_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    cuda_wheel_urls: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Emit a pixi `[feature.<name>.*]` block from a node's ComfyEnvConfig.

    Only carries node-specific deps. Python pin lives in the pyXY feature; torch
    lives in the comfyui feature. No platform gates -- pypi index resolution
    handles wheel selection. Plain torch/torchvision/torchaudio entries are
    stripped from `[dependencies]`/`[pypi-dependencies]` (workspace-global from
    the `comfyui` feature).

    `pypi-options` from the node's `comfy-env.toml` is passed through verbatim.
    A node author can express `[pypi-options.dependency-overrides]` to redirect
    torch (or any package) resolution within their per-node env independently
    of the comfyui workspace pin.

    `auto_overrides`, when given, is a `{pkg: {version, index}}` map populated
    by `build_workspace_toml` when the resolved cuda-wheel combo diverges from
    bootstrap. It's merged into `pypi-options.dependency-overrides`, with any
    manual override from the node's `comfy-env.toml` winning on conflict.
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

    if cuda_wheel_urls:
        pypi = feat.setdefault("pypi-dependencies", {})
        for pkg, url in cuda_wheel_urls.items():
            pypi[pkg] = {"url": url}
        log(
            f"[comfy-env] {name}: cuda-wheels inlined as pypi-dependencies "
            f"({', '.join(cuda_wheel_urls.keys())})"
        )

    pypi_options = copy.deepcopy(cfg.pixi_passthrough.get("pypi-options", {}))
    if auto_overrides:
        manual = pypi_options.get("dependency-overrides", {}) or {}
        merged = {**auto_overrides, **manual}  # manual entries shadow auto-emitted
        if merged:
            pypi_options["dependency-overrides"] = merged
        # one-line summary instead of three per node
        any_spec = next(iter(auto_overrides.values()), None)
        idx = (any_spec or {}).get("index", "")
        ver_summary = ", ".join(
            f"{p}{(s or {}).get('version', '')}" for p, s in auto_overrides.items()
        )
        shadowed = sorted(set(auto_overrides) & set(manual))
        shadow_note = f" (overridden by comfy-env.toml: {', '.join(shadowed)})" if shadowed else ""
        log(
            f"[comfy-env] {name}: torch override -> {ver_summary} from {idx}"
            f"{shadow_note}"
        )
    if pypi_options:
        feat["pypi-options"] = pypi_options

    return feat


def _torch_family_pins(
    torch_pin: Optional[str],
    log: Callable[[str], None],
) -> Optional[Dict[str, str]]:
    """Return {torch: <pin>, torchvision: <pin>, torchaudio: <pin>} from TORCH_FAMILY_COMPAT.

    Returns None when `torch_pin` is None. Returns a partial map (`{"torch": pin}` only)
    when torch's minor isn't in the compat table, with a warning logged.
    """
    if not torch_pin:
        return None
    from .cuda_wheels import derive_family_pins
    family = derive_family_pins(torch_pin)
    if family is None:
        log(
            f"[comfy-env] WARNING: torch_pin {torch_pin} not in TORCH_FAMILY_COMPAT; "
            f"pinning torch only, leaving torchvision/torchaudio unpinned"
        )
        return {"torch": torch_pin}
    vision_pin, audio_pin = family
    return {"torch": torch_pin, "torchvision": vision_pin, "torchaudio": audio_pin}


def _pin_torch_family(
    pypi: Dict[str, Any],
    torch_pin: Optional[str],
    log: Callable[[str], None],
) -> None:
    """Pin torch and derive matching torchvision/torchaudio pins in a pypi dict.

    `torch_pin` is a PEP 440 spec like `"==2.11.0"` (tier-1) or `"==2.8.*"` (tier-2
    fallback). The torch INDEX (cu{NN} / cpu) was attached upstream by
    parse_comfyui_requirements; here we just clamp versions so per-node envs
    hardlink-share an identical torch family with the comfyui template env.

    No-op when `torch_pin` is None.
    """
    pin_map = _torch_family_pins(torch_pin, log)
    if not pin_map:
        return

    pinned: list[str] = []
    for k in list(pypi.keys()):
        new_spec = pin_map.get(k.lower())
        if new_spec is None:
            continue
        spec = pypi[k]
        if isinstance(spec, dict):
            pypi[k] = {**spec, "version": new_spec}
        else:
            pypi[k] = new_spec
        pinned.append(f"{k} {new_spec}")
    if pinned:
        log(f"[comfy-env] Comfyui feature pin: {', '.join(pinned)}")


_PYTORCH_PACKAGES = frozenset({"torch", "torchvision", "torchaudio"})


def build_workspace_toml(
    comfyui_dir: Path,
    torch_index: Optional[str],
    cuda_major: Optional[str],
    node_configs: List[Tuple[str, ComfyEnvConfig]],  # (env_name, cfg) pairs
    bootstrap_python: Optional[str] = None,
    torch_pin: Optional[str] = None,
    log: Callable[[str], None] = print,
    chosen_torch_index: Optional[str] = None,
    chosen_torch_pin: Optional[str] = None,
    chosen_cuda: Optional[str] = None,
    chosen_torch_short: Optional[str] = None,
    chosen_python: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the full workspace pixi.toml as a dict.

    Args:
        comfyui_dir: ComfyUI install root (used to find requirements.txt).
        torch_index: PyPI index URL for the comfyui feature's torch (bootstrap-derived,
            i.e. what python_embeded already has). The comfyui template env is meant
            to mirror python_embeded; per-node envs that need a different cuda combo
            override below.
        cuda_major: Major CUDA version for `[system-requirements]` (e.g. "12"). None on CPU.
        node_configs: List of (env_name, ComfyEnvConfig) -- one entry per environment to create.
        bootstrap_python: Major.minor of the install bootstrap interpreter (e.g. "3.10").
            Defaults to `sys.version_info` on the running interpreter.
        torch_pin: PEP 440 version spec to pin torch/torchvision/torchaudio in the
            comfyui feature (bootstrap-derived). When None, torch stays as `*`.
        chosen_torch_index, chosen_torch_pin: the cuda-wheel combo selected by
            `_resolve_wheel_combo` -- may equal the bootstrap (no real divergence)
            or differ (fallback combo). When set, per-node features that declare
            cuda-only packages get a `[pypi-options.dependency-overrides]` block
            targeting this combo so their env resolves to a torch matching the
            cuda-only wheels, independent of the comfyui pin.
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
    # Conda-forge MKL pulls Intel libiomp5md.dll; pip-installed torch ships
    # LLVM libomp.dll. With both loaded in the same process, torch's OMP
    # guard aborts ("OMP: Error #15 ... already initialized"). Set on the
    # comfyui feature so every env (template + per-node) inherits it; OMP
    # documents this var as the safe escape when full libomp dedupe across
    # mixed conda/pip wheels isn't feasible.
    feature_comfyui["activation"] = {"env": {"KMP_DUPLICATE_LIB_OK": "TRUE"}}
    out["feature"] = {"comfyui": feature_comfyui}

    # Per-python features
    py_versions: Dict[str, str] = {host_py: _python_feature_name(host_py)}
    for _, cfg in node_configs:
        v = cfg.python or host_py
        py_versions.setdefault(v, _python_feature_name(v))
    for v, fname in py_versions.items():
        out["feature"][fname] = _build_python_feature(v)

    # Per-node features.
    # Build the override map once from the chosen cuda-wheel combo (same combo
    # workspace-wide today; one entry per torch-family package). `_build_node_feature`
    # only attaches it to features whose node declares cuda-only packages — no-cuda
    # nodes get no auto-emit. When chosen == bootstrap the override is redundant but
    # explicit; when they diverge it's the wire that lets the per-node env resolve a
    # different torch than the comfyui template env.
    override_map: Optional[Dict[str, Dict[str, str]]] = None
    if chosen_torch_pin and chosen_torch_index:
        family = _torch_family_pins(chosen_torch_pin, log)
        if family:
            override_map = {
                pkg: {"version": pin, "index": chosen_torch_index}
                for pkg, pin in family.items()
            }
            log(
                f"[comfy-env] cuda-wheels combo: per-node cuda features will pin "
                f"{sorted(family.keys())} via pypi-options.dependency-overrides "
                f"({chosen_torch_index})"
            )

    # Resolve cuda-wheel URLs for each per-node feature so pixi installs them
    # as part of `pixi install --all` (rather than a slow post-step pip pass).
    from .cuda_wheels import get_wheel_url as _get_wheel_url
    can_resolve_urls = bool(chosen_cuda and chosen_torch_short and chosen_python)

    for env_name, cfg in node_configs:
        cuda_only = [p for p in cfg.cuda_packages if p not in _PYTORCH_PACKAGES]
        feat_urls: Optional[Dict[str, str]] = None
        if cuda_only and can_resolve_urls:
            urls: Dict[str, str] = {}
            for pkg in cuda_only:
                url = _get_wheel_url(
                    pkg, chosen_torch_short, chosen_cuda, chosen_python, log=log,
                )
                if not url:
                    raise RuntimeError(
                        f"cuda-wheel {pkg!r} unavailable for "
                        f"cu{chosen_cuda}/torch{chosen_torch_short}/cp{chosen_python}; "
                        f"_resolve_wheel_combo should have caught this earlier."
                    )
                urls[pkg] = url
            feat_urls = urls

        feat = _build_node_feature(
            cfg, env_name, log,
            auto_overrides=override_map if cuda_only else None,
            cuda_wheel_urls=feat_urls,
        )
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
    chosen_torch_index: Optional[str] = None,
    chosen_torch_pin: Optional[str] = None,
    chosen_cuda: Optional[str] = None,
    chosen_torch_short: Optional[str] = None,
    chosen_python: Optional[str] = None,
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
        chosen_torch_index=chosen_torch_index,
        chosen_torch_pin=chosen_torch_pin,
        chosen_cuda=chosen_cuda,
        chosen_torch_short=chosen_torch_short,
        chosen_python=chosen_python,
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
