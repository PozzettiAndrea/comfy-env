"""Generate the workspace pixi.toml from ComfyUI requirements + per-node configs.

Workspace model:
- One pixi workspace per ComfyUI install at `<comfyui_dir>/.ce/pixi.toml`.
- One self-contained `[feature.<env_name>]` per environment. Each carries its
  own python pin, pip/setuptools, glibc, KMP env var, torch family pin, and
  declared deps. Nothing is shared between features.
- `[environments]` maps `<env_name> -> [<env_name>]` with `no-default-feature = true`.
  No solve-groups: every env solves independently.
- The only cross-env coupling is the torch pin, which is replicated verbatim
  into each feature so workers and parent share an identical torch family.
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

def _common_base_dependencies(version: str) -> Dict[str, Any]:
    """Conda deps every env needs: python pin + pip + setuptools."""
    return {
        "python": f"{version}.*",
        "pip": "*",
        "setuptools": ">=75.0,<82",
    }


def _torch_family_pypi(
    torch_pin: Optional[str],
    torch_index: Optional[str],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """Build `{torch, torchvision, torchaudio}` pypi-deps with `index` attached.

    Replicated verbatim into every feature so each env resolves identical torch
    files and pixi's content-addressable cache hardlink-shares them. Returns {}
    on CPU/macOS hosts where there's no workspace-wide pin.
    """
    if not torch_pin or not torch_index:
        return {}
    pin_map = _torch_family_pins(torch_pin, log)
    if not pin_map:
        return {}
    return {pkg: {"version": pin, "index": torch_index}
            for pkg, pin in pin_map.items()}


def _validate_node_config(name: str, cfg: ComfyEnvConfig) -> None:
    """Reject node configs that try to redefine the workspace torch pin."""
    bad = [p for p in cfg.cuda_packages if p in _TORCH_PKGS]
    if bad:
        raise ValueError(
            f"[{name}] comfy-env.toml has {bad} under [cuda] packages. "
            "Plain torch/torchvision/torchaudio are pinned workspace-wide "
            "(replicated into every feature so the rattler cache dedupes). "
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

    Torch family is pinned workspace-wide and added directly to each feature by
    `_build_node_feature`. Node-level declarations are stripped so they can't
    shadow the pin.
    """
    for k in list(table.keys()):
        if k.lower() in _TORCH_PKGS:
            del table[k]
            log(f"[comfy-env] {name}: ignoring `{k}` in {where} (pinned workspace-wide)")


def _build_node_feature(
    cfg: ComfyEnvConfig,
    name: str,
    version: str,
    torch_pin: Optional[str],
    torch_index: Optional[str],
    glibc_version: Optional[str],
    log: Callable[[str], None] = print,
    cuda_wheel_urls: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Emit a self-contained pixi `[feature.<name>.*]` block for one env.

    Each node env owns its full feature: python pin, pip/setuptools, glibc,
    KMP env var, torch family pin (replicated from workspace -- see
    `_torch_family_pypi`), and the node's own deps from `comfy-env.toml`.
    No base feature is composed in. Plain torch/torchvision/torchaudio entries
    in the node's own deps are stripped so they can't shadow the workspace pin.

    `pypi-options` from the node's `comfy-env.toml` is passed through verbatim,
    so a node author can still express `[pypi-options.dependency-overrides]`
    manually to redirect a specific transitive resolution.
    """
    feat: Dict[str, Any] = {}

    # Conda deps: base (python/pip/setuptools) + node's own
    deps = _common_base_dependencies(version)
    node_conda = copy.deepcopy(cfg.pixi_passthrough.get("dependencies", {}))
    if node_conda:
        _strip_torch_family(node_conda, name, "[dependencies]", log)
        deps.update(node_conda)
    feat["dependencies"] = deps

    # PyPI deps: torch family pin (replicated workspace-wide) + node's own
    pypi = _torch_family_pypi(torch_pin, torch_index, log)
    node_pypi = copy.deepcopy(cfg.pixi_passthrough.get("pypi-dependencies", {}))
    if node_pypi:
        _strip_torch_family(node_pypi, name, "[pypi-dependencies]", log)
        pypi.update(node_pypi)
    if pypi:
        feat["pypi-dependencies"] = pypi

    # Per-target sections (only the current platform's), with torch family stripped
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
    if pypi_options:
        feat["pypi-options"] = pypi_options

    # System requirements: node-declared wins, else workspace glibc
    sys_reqs = cfg.pixi_passthrough.get("system-requirements")
    if sys_reqs:
        feat["system-requirements"] = copy.deepcopy(sys_reqs)
    elif glibc_version:
        feat["system-requirements"] = {
            "libc": {"family": "glibc", "version": glibc_version},
        }

    feat["activation"] = {"env": {"KMP_DUPLICATE_LIB_OK": "TRUE"}}
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
    comfyui_source_dir: Optional[Path] = None,
    log: Callable[[str], None] = print,
    chosen_torch_index: Optional[str] = None,
    chosen_torch_pin: Optional[str] = None,
    chosen_cuda: Optional[str] = None,
    chosen_torch_short: Optional[str] = None,
    chosen_python: Optional[str] = None,
    root_conda_deps: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the full workspace pixi.toml as a dict.

    Each environment gets exactly one self-contained `[feature.<env_name>]`:
    its own python pin, pip/setuptools, glibc, KMP env, torch family pin, and
    declared deps. No shared base feature, no solve-group — every env solves
    independently. Per-env cross-coupling is limited to the workspace torch
    pin, which is replicated verbatim into each feature so the rattler/uv
    cache hardlink-shares torch's binary payload across envs.

    Per-env defaults that fall back to the workspace baseline (a node may
    override any of these in its `comfy-env.toml`):
      - python: `cfg.python` or `host_py` (bootstrap_python).
      - torch family: bootstrap torch_pin/torch_index, except cuda nodes use
        `chosen_torch_pin`/`chosen_torch_index` from the cuda-wheel resolver
        (these equal bootstrap unless the resolver had to fall back).
      - glibc: auto-detected from host, used only when the node didn't
        declare its own `[system-requirements]`.
    """
    host_py = bootstrap_python or f"{sys.version_info.major}.{sys.version_info.minor}"
    current_platform = get_pixi_platform()

    # Validate first so we fail fast on bad node configs
    for name, cfg in node_configs:
        _validate_node_config(name, cfg)

    # Collect channels: start with conda-forge, then merge any extra channels
    # declared by node configs in [workspace].channels.
    channels: List[str] = ["conda-forge"]
    for _, cfg in node_configs:
        for ch in cfg.pixi_passthrough.get("workspace", {}).get("channels", []):
            if ch not in channels:
                channels.append(ch)

    workspace: Dict[str, Any] = {
        "name": "comfy-env",
        "version": "0.1.0",
        "channels": channels,
        "platforms": [current_platform],
    }

    out: Dict[str, Any] = {"workspace": workspace}

    # Auto-detect host glibc
    import platform as _platform
    libc_family, libc_version = _platform.libc_ver()
    glibc_version: Optional[str] = None
    if libc_family == "glibc" and libc_version:
        glibc_version = libc_version
        log(f"[comfy-env] Host glibc {libc_version} -> system-requirements")

    out["feature"] = {}

    # Resolve cuda-wheel URLs for each cuda node so pixi installs them as part
    # of `pixi install --all` (rather than a slow post-step pip pass).
    from .cuda_wheels import get_wheel_url as _get_wheel_url
    can_resolve_urls = bool(chosen_cuda and chosen_torch_short)

    # Collect cuda-wheel URLs per env for post-pixi `uv pip install --no-deps`.
    # These are NOT inlined into pixi.toml because pixi's resolver cannot handle
    # --no-deps semantics and will try to resolve/build their declared dependencies.
    cuda_urls_by_env: Dict[str, Dict[str, str]] = {}

    for env_name, cfg in node_configs:
        cuda_only = [p for p in cfg.cuda_packages if p not in _PYTORCH_PACKAGES]
        env_python = cfg.python or host_py

        if cuda_only and can_resolve_urls and env_python:
            urls: Dict[str, str] = {}
            for pkg in cuda_only:
                url = _get_wheel_url(
                    pkg, chosen_torch_short, chosen_cuda, env_python, log=log,
                )
                if not url:
                    raise RuntimeError(
                        f"cuda-wheel {pkg!r} unavailable for "
                        f"cu{chosen_cuda}/torch{chosen_torch_short}/cp{env_python}; "
                        f"_resolve_wheel_combo should have caught this earlier."
                    )
                urls[pkg] = url
            if urls:
                cuda_urls_by_env[env_name] = urls
                log(
                    f"[comfy-env] {env_name}: cuda-wheels deferred for post-pixi install "
                    f"({', '.join(urls.keys())})"
                )

        # Cuda nodes use the chosen combo (may differ from bootstrap when the
        # resolver fell back). Non-cuda nodes use bootstrap torch directly.
        if cuda_only and chosen_torch_pin and chosen_torch_index:
            node_torch_pin: Optional[str] = chosen_torch_pin
            node_torch_index: Optional[str] = chosen_torch_index
        else:
            node_torch_pin = torch_pin
            node_torch_index = torch_index

        out["feature"][env_name] = _build_node_feature(
            cfg, env_name, env_python,
            torch_pin=node_torch_pin,
            torch_index=node_torch_index,
            glibc_version=glibc_version,
            log=log,
        )

    # Environments table. One env -> one self-contained feature, no solve-group.
    # Every env solves independently; the only cross-env coupling is the torch
    # pin replicated into each feature.
    environments: Dict[str, Any] = {}
    for env_name, _ in node_configs:
        environments[env_name] = {
            "features": [env_name],
            "no-default-feature": True,
        }
    out["environments"] = environments

    return out, cuda_urls_by_env


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
    root_conda_deps: Optional[Dict[str, Any]] = None,
    comfyui_source_dir: Optional[Path] = None,
) -> Path:
    """Generate `<workspace_dir>/pixi.toml` from the parts above.

    Returns (pixi_toml_path, cuda_urls_by_env) where cuda_urls_by_env is
    ``{env_name: {pkg: wheel_url}}`` for post-pixi ``uv pip install --no-deps``.
    """
    tomli_w = _require_tomli_w()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    pixi_toml = workspace_dir / "pixi.toml"
    data, cuda_urls_by_env = build_workspace_toml(
        comfyui_dir, torch_index, cuda_major, node_configs,
        bootstrap_python=bootstrap_python,
        torch_pin=torch_pin,
        comfyui_source_dir=comfyui_source_dir,
        log=log,
        chosen_torch_index=chosen_torch_index,
        chosen_torch_pin=chosen_torch_pin,
        chosen_cuda=chosen_cuda,
        chosen_torch_short=chosen_torch_short,
        chosen_python=chosen_python,
        root_conda_deps=root_conda_deps,
    )

    # The workspace is shared across every ComfyUI install on this machine
    # (conda-style global env pool). Merge this install's features and
    # environments on top of whatever was there from prior runs so we don't
    # delete envs that belong to other installs.
    if pixi_toml.exists():
        import tomli
        try:
            with open(pixi_toml, "rb") as f:
                existing = tomli.load(f)
        except Exception as e:
            log(f"[comfy-env] Warning: couldn't read existing {pixi_toml} ({e}); overwriting.")
            existing = None
        if existing:
            data = _merge_into_existing(existing, data)

    with open(pixi_toml, "wb") as f:
        tomli_w.dump(data, f)
    log(f"Generated {pixi_toml}")
    return pixi_toml, cuda_urls_by_env


def _merge_into_existing(existing: Dict[str, Any], fresh: Dict[str, Any]) -> Dict[str, Any]:
    """Merge `fresh` (this install's envs) on top of `existing` (prior installs).

    - `feature.<name>` and `environments.<name>` entries from `fresh` overwrite
      same-named entries in `existing`; entries only in `existing` are kept.
    - `workspace.channels` is unioned (fresh-preferred order).
    - Everything else under `workspace.*` comes from `fresh` (name/platforms/version
      are identical across installs on the same machine).
    """
    merged = copy.deepcopy(existing)
    fresh_ws = fresh.get("workspace", {})
    old_ws = existing.get("workspace", {})
    new_channels = list(fresh_ws.get("channels", []))
    old_channels = list(old_ws.get("channels", []))
    seen: set = set()
    union: List[str] = []
    for c in new_channels + old_channels:
        if c not in seen:
            seen.add(c)
            union.append(c)
    merged_ws = dict(fresh_ws)
    if union:
        merged_ws["channels"] = union
    merged["workspace"] = merged_ws
    merged.setdefault("feature", {}).update(fresh.get("feature", {}))
    merged.setdefault("environments", {}).update(fresh.get("environments", {}))
    return merged


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
