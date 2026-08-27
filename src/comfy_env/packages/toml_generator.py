"""Generate the workspace pixi.toml from ComfyUI requirements + per-node configs.

Workspace model:
- One manifest per env, each a standalone pixi workspace. It carries its own
  python pin, pip/setuptools, glibc, KMP env var, torch family pin and deps;
  nothing is shared, and a parse error in one env cannot affect another.
- The feature is always named `node` and the environment always `default`
  (pixi reserves `default` as a feature name -- see build_env_toml).
- The only cross-env coupling is the torch pin, which is replicated verbatim
  into each feature so workers and parent share an identical torch family.
"""

import copy
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config import ComfyEnvConfig
from ..detection import get_pixi_platform


_TORCH_PKGS = {"torch", "torchvision", "torchaudio"}


def _require_tomli_w():
    try:
        import tomli_w
        return tomli_w
    except ImportError:
        raise ImportError("tomli-w required: pip install tomli-w")


# requirements.txt parsing


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
    """Build `{torch, torchvision, torchaudio}` pypi-deps, optionally with `index` attached.

    Replicated verbatim into every feature so each env resolves identical torch
    files and pixi's content-addressable cache hardlink-shares them. When
    `torch_index` is set (Linux/Windows + CUDA path), the index is attached to
    every entry. When it's None (macOS, where the PyTorch /whl/cpu index has no
    osx-arm64 wheels), the pin is emitted without an index so pixi resolves
    from the default PyPI. `_torch_family_pins` already returns a partial map
    `{"torch": pin}` when the family table doesn't know `torch.minor` yet --
    that's the right behavior for torch 2.12-style "ahead of family" cases
    where torchaudio hasn't shipped yet.
    """
    if not torch_pin:
        return {}
    pin_map = _torch_family_pins(torch_pin, log)
    if not pin_map:
        return {}
    if torch_index:
        return {pkg: {"version": pin, "index": torch_index}
                for pkg, pin in pin_map.items()}
    return dict(pin_map)


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


# ---------------------------------------------------------------------------
# ADR-0013 passthrough contract: forward everything pixi owns; the sets below
# are the ENTIRE schema knowledge comfy-env keeps about pixi's language.
#
# _HANDLED_PASSTHROUGH: tables the compiler already places/transforms itself
#   (torch-family REWRITE inside dependencies/pypi-dependencies; [activation]
#   MERGE; workspace.channels forwarding). Wire types declare in
#   comfy-env-root.toml [types] (ADR-0015), not here.
# _DENIED_PASSTHROUGH / _DENIED_WORKSPACE_KEYS: compiler-owned, hard error --
#   the single-feature/single-env manifest shape IS ADR-0007, and platforms/
#   name/version are host-derived identity.
# Everything else is forwarded verbatim at the feature level; pixi (pinned,
# ADR-0002) validates its own language.
# ---------------------------------------------------------------------------
_HANDLED_PASSTHROUGH = {
    "dependencies", "pypi-dependencies", "target", "pypi-options",
    "system-requirements", "activation", "workspace",
}
_DENIED_PASSTHROUGH = {"environments", "feature"}
_DENIED_WORKSPACE_KEYS = {"name", "version", "platforms"}


def _validate_passthrough(env_name: str, cfg: ComfyEnvConfig) -> None:
    """Hard-error on compiler-owned keys (ADR-0013 deny list)."""
    denied = sorted(_DENIED_PASSTHROUGH & set(cfg.pixi_passthrough))
    if denied:
        raise ValueError(
            f"{env_name}: comfy-env.toml declares compiler-owned table(s) "
            f"{', '.join('[' + d + ']' for d in denied)} -- the per-env "
            f"manifest is single-feature/single-environment by design "
            f"(ADR-0007); comfy-env generates these."
        )
    ws = cfg.pixi_passthrough.get("workspace", {})
    ws_denied = sorted(_DENIED_WORKSPACE_KEYS & set(ws))
    if ws_denied:
        raise ValueError(
            f"{env_name}: comfy-env.toml declares compiler-owned workspace "
            f"key(s) {', '.join(ws_denied)} -- name/version are env identity "
            f"and platforms is derived from the host machine."
        )


def _build_node_feature(
    cfg: ComfyEnvConfig,
    name: str,
    version: str,
    torch_pin: Optional[str],
    torch_index: Optional[str],
    glibc_version: Optional[str],
    log: Callable[[str], None] = print,
    macos_version: Optional[str] = None,
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
    # CUDA wheels, inlined as direct-URL deps (may carry a #sha256= fragment,
    # which pixi records in the lock and uv verifies). The wheels' in-farm
    # METADATA declares no dependencies, so a URL dep is exactly the
    # `--no-deps` semantics the retired post-pixi uv pass had -- but inside
    # pixi.lock, hashed, cached, and safe against a plain `pixi install`.
    # Merged LAST so neither the torch pin nor an author entry shadows it.
    if cuda_wheel_urls:
        for _pkg, _url in sorted(cuda_wheel_urls.items()):
            _key = _pkg.lower().replace("_", "-")
            pypi[_key] = {"url": _url}
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

    pypi_options = copy.deepcopy(cfg.pixi_passthrough.get("pypi-options", {}))
    if pypi_options:
        feat["pypi-options"] = pypi_options

    # System requirements: node-declared wins outright; else auto-detect from
    # host (glibc on Linux, torch's wheel macOS floor on macOS). Both may be
    # set on the same platform if applicable.
    sys_reqs = cfg.pixi_passthrough.get("system-requirements")
    if sys_reqs:
        feat["system-requirements"] = copy.deepcopy(sys_reqs)
    else:
        auto_reqs: Dict[str, Any] = {}
        if glibc_version:
            auto_reqs["libc"] = {"family": "glibc", "version": glibc_version}
        if macos_version:
            auto_reqs["macos"] = macos_version
        if auto_reqs:
            feat["system-requirements"] = auto_reqs

    # [activation]: MERGE (ADR-0013) -- author entries pass through; the
    # compiler's own key wins only on direct collision. Assigning the block
    # outright silently clobbers author activation.
    activation = copy.deepcopy(cfg.pixi_passthrough.get("activation", {}))
    if not isinstance(activation, dict):
        activation = {}
    act_env = activation.setdefault("env", {})
    act_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    feat["activation"] = activation

    # Honest passthrough (ADR-0013): every remaining table is forwarded
    # verbatim at the feature level ([tasks], future pixi tables, ...).
    # pixi validates its own language; comfy-env does not chase its schema.
    for key, value in cfg.pixi_passthrough.items():
        if key in _HANDLED_PASSTHROUGH or key in _DENIED_PASSTHROUGH:
            continue
        feat[key] = copy.deepcopy(value)
        log(f"[comfy-env] {name}: passthrough [{key}] -> feature")
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



def build_env_toml(
    env_name: str,
    cfg: ComfyEnvConfig,
    torch_index: Optional[str],
    bootstrap_python: Optional[str] = None,
    torch_pin: Optional[str] = None,
    chosen_torch_index: Optional[str] = None,
    chosen_torch_pin: Optional[str] = None,
    log: Callable[[str], None] = print,
    cuda_wheel_urls: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build a self-contained pixi.toml dict for one isolated env.

    Each env gets its own manifest declaring a single feature ``node``
    and a single environment ``default``. No solve-groups, no cross-env
    references. A parse error in one env's pixi.toml has zero effect on
    any other env.

    Args:
        env_name: Logical env name (used in workspace/name only, not as the
            pixi environment name -- that's always ``default``).
        cfg: Node's parsed comfy-env.toml.
        torch_index: Workspace-wide bootstrap torch index URL (cu*/cpu).
        torch_pin: Workspace-wide bootstrap torch version pin (``==X.Y.Z``).
        chosen_*: Override combo for cuda-only nodes (from cuda-wheel resolver).
    """
    host_py = bootstrap_python or f"{sys.version_info.major}.{sys.version_info.minor}"
    current_platform = get_pixi_platform()

    _validate_node_config(env_name, cfg)
    _validate_passthrough(env_name, cfg)

    # Workspace-section channels: conda-forge + any extras from this node's config
    channels: List[str] = ["conda-forge"]
    for ch in cfg.pixi_passthrough.get("workspace", {}).get("channels", []):
        if ch not in channels:
            channels.append(ch)

    # Auto-detect host glibc.
    import platform as _platform
    libc_family, libc_version = _platform.libc_ver()
    glibc_version: Optional[str] = None
    if libc_family == "glibc" and libc_version:
        glibc_version = libc_version

    # Auto-detect torch's macOS wheel floor. pixi defaults osx-arm64 to
    # macOS 13, but torch 2.12+ only ships macosx_14_0_arm64 wheels — so
    # without this, pixi solve fails with "no matching platform tag".
    macos_version: Optional[str] = None
    if sys.platform == "darwin":
        from ..detection.cuda import get_bootstrap_torch_macos_min
        macos_version = get_bootstrap_torch_macos_min()
        if macos_version:
            log(f"[comfy-env] Torch wheel targets macOS {macos_version}+ "
                f"-> emitting system-requirements macos={macos_version}")

    # cuda-wheel-only nodes use the override combo; everyone else gets bootstrap.
    cuda_only = [p for p in cfg.cuda_packages if p not in _TORCH_PKGS]
    if cuda_only and chosen_torch_pin and chosen_torch_index:
        node_torch_pin: Optional[str] = chosen_torch_pin
        node_torch_index: Optional[str] = chosen_torch_index
    else:
        node_torch_pin = torch_pin
        node_torch_index = torch_index

    env_python = cfg.python or host_py
    # Feature name MUST NOT be "default" -- pixi reserves that name for the
    # implicit feature that picks up any tables at the toml root, and refuses
    # to parse a manifest that declares `[feature.default.*]` explicitly.
    # Use "node" instead; the user-facing pixi environment name stays "default".
    feat = _build_node_feature(
        cfg, env_name, env_python,
        torch_pin=node_torch_pin,
        torch_index=node_torch_index,
        glibc_version=glibc_version,
        macos_version=macos_version,
        log=log,
        cuda_wheel_urls=cuda_wheel_urls,
    )

    return {
        "workspace": {
            "name": f"comfy-env-{env_name}",
            "version": "0.1.0",
            "channels": channels,
            "platforms": [current_platform],
        },
        "feature": {"node": feat},
        "environments": {
            "default": {"features": ["node"], "no-default-feature": True},
        },
    }


def write_env_pixi_toml(
    env_manifest_dir: Path,
    env_name: str,
    cfg: ComfyEnvConfig,
    torch_index: Optional[str],
    bootstrap_python: Optional[str] = None,
    torch_pin: Optional[str] = None,
    chosen_torch_index: Optional[str] = None,
    chosen_torch_pin: Optional[str] = None,
    log: Callable[[str], None] = print,
    cuda_wheel_urls: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Write ``<env_manifest_dir>/pixi.toml`` for one isolated env.

    Returns the manifest DICT (used for derivation-identity hashing).
    Overwrites any existing file -- per-env manifests are deterministically
    regenerated from the node's config.
    """
    tomli_w = _require_tomli_w()
    env_manifest_dir.mkdir(parents=True, exist_ok=True)
    pixi_toml = env_manifest_dir / "pixi.toml"
    data = build_env_toml(
        env_name=env_name,
        cfg=cfg,
        torch_index=torch_index,
        bootstrap_python=bootstrap_python,
        torch_pin=torch_pin,
        chosen_torch_index=chosen_torch_index,
        chosen_torch_pin=chosen_torch_pin,
        log=log,
        cuda_wheel_urls=cuda_wheel_urls,
    )
    # Provenance header (ADR-0013): when pixi rejects a forwarded key its
    # error names THIS generated file, which the author never wrote -- the
    # header leads the trail back to the real source.
    from .. import __version__ as _ce_version
    header = (
        f"# Generated by comfy-env {_ce_version} -- DO NOT EDIT.\n"
        f"# Source: the comfy-env.toml of env '{env_name}'. Fix errors there.\n"
    )
    with open(pixi_toml, "wb") as f:
        f.write(header.encode("utf-8"))
        tomli_w.dump(data, f)
    return data


def resolve_env_cuda_wheel_urls(
    cfg: ComfyEnvConfig,
    bootstrap_python: Optional[str],
    chosen_cuda: Optional[str],
    chosen_torch_short: Optional[str],
    log: Callable[[str], None] = print,
) -> Dict[str, str]:
    """Return the cuda-wheel URLs for one env, keyed by declared package name.

    The URLs are inlined into the generated manifest as direct-URL
    pypi-dependencies (``build_env_toml(cuda_wheel_urls=...)``), so they land
    in pixi.lock. They may carry ``#sha256=`` fragments; preserve them.
    """
    cuda_only = [p for p in cfg.cuda_packages if p not in _TORCH_PKGS]
    if not (cuda_only and chosen_cuda and chosen_torch_short):
        return {}
    env_python = cfg.python or bootstrap_python
    if not env_python:
        return {}

    from .cuda_wheels import get_wheel_url as _get_wheel_url
    urls: Dict[str, str] = {}
    for pkg in cuda_only:
        url = _get_wheel_url(pkg, chosen_torch_short, chosen_cuda, env_python, log=log)
        if not url:
            raise RuntimeError(
                f"cuda-wheel {pkg!r} unavailable for "
                f"cu{chosen_cuda}/torch{chosen_torch_short}/"
                f"cp{env_python.replace('.', '')}; "
                f"_resolve_wheel_combo should have caught this earlier."
            )
        urls[pkg] = url
    return urls


