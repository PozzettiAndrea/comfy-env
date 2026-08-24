"""Environment path resolution for comfy-env.

Layout (per-env manifests, v0.4+):

    <workspace_root>/
      envs/
        <env_name>/
          pixi.toml                  # one env per file
          pixi.lock
          .pixi/envs/default/        # materialized env

Each env's manifest is isolated -- a parse error in one cannot poison
another. `<workspace_root>` is shared machine-wide so two ComfyUI installs
that declare the same node reuse the same materialized env dir (cross-install
sharing).

No backward compatibility with the v0.3.x single-file layout. Workspaces
created by v0.3.x (``<workspace>/pixi.toml`` + ``<workspace>/.pixi/envs/<name>/``)
are invisible to v0.4+; they need to be re-materialized via
``comfy-env install``. User is expected to ``rm -rf`` the legacy
``<workspace>/.pixi/`` and ``<workspace>/pixi.toml`` once they've moved over.
"""

import os
import re
import shutil
import sys
from pathlib import Path

# Legacy workspace dir name (kept as a constant for the one orphan-detect
# log line in `install_workspace`).
CE_WORKSPACE_DIR = ".ce"


_ANNOUNCED_WS = False


def _sanitize_pixi_name(s: str) -> str:
    """Pixi environment names must match `[a-z0-9-]+`. Collapse any other
    char run to a single dash and trim, so a folder rename like `Foo._disabled`
    or a name containing dots / parens / spaces cannot poison a manifest.
    """
    s = re.sub(r"[^a-z0-9-]+", "-", s.lower())
    return re.sub(r"-+", "-", s).strip("-")


def _windows_local_appdata() -> Path:
    """Real user's LOCALAPPDATA, never the systemprofile one.

    Same hazard `_candidate_config_dirs` already guards against, but it only
    guarded ComfyUI Desktop config discovery -- the workspace root read
    LOCALAPPDATA raw. Under a SYSTEM/service/scheduled-task shell that resolves
    to C:\\Windows\\System32\\config\\systemprofile\\AppData\\Local, so a whole
    SECOND machine-global store gets built there. That defeats the entire point
    of this root, which exists so installs SHARE one multi-GB materialized env
    (see module docstring); two roots share nothing.

    Observed on this box: USERNAME=ANDREW-PC$ (a machine account),
    LOCALAPPDATA under systemprofile, while USERPROFILE=C:\\Users\\andrew was
    correct -- hence the USERPROFILE fallback before the C:\\Users\\* glob.
    """
    local = os.environ.get("LOCALAPPDATA", "")
    if local and "systemprofile" not in local.lower():
        return Path(local)

    userprofile = os.environ.get("USERPROFILE", "")
    if userprofile and "systemprofile" not in userprofile.lower():
        return Path(userprofile) / "AppData" / "Local"

    username = os.environ.get("USERNAME", "")
    # Reject SYSTEM and machine accounts (trailing '$'), which have no profile
    # of their own. _candidate_config_dirs does not cover the '$' case.
    if username and username.upper() != "SYSTEM" and not username.endswith("$"):
        return Path("C:/Users") / username / "AppData" / "Local"

    try:
        from glob import glob as _glob
        skip = ("default", "default user", "public", "all users")
        for p in _glob(r"C:\Users\*\AppData\Local"):
            parts = Path(p).parts
            if len(parts) > 2 and parts[2].lower() in skip:
                continue
            return Path(p)
    except Exception:
        pass

    # Nothing usable: fall back to the raw value rather than inventing a path,
    # so the failure is visible in the announced workspace line.
    return Path(local) if local else Path.home() / "AppData" / "Local"


_ABI_TAG = None


def _abi_tag():
    """ABI identity of the bootstrap interpreter, e.g. ``py313-torch2.10-cu128``.

    The workspace root is shared machine-wide and envs were keyed on the node
    name ALONE, but a materialized env is not interchangeable across stacks:
    its manifest pins ``python = "3.13.*"`` and ``torch == "2.10.0"`` from
    whichever ComfyUI happened to run the install, and its cuda-only wheels are
    stamped with the same (``cumesh-0.0.1+cu128torch2.10``).

    So two ComfyUI installs on different stacks both resolved to
    ``envs/<node>/`` and either

      * thrashed -- the second install regenerated the shared pixi.toml from
        its own bootstrap and re-materialized the whole multi-GB env, and
        switching back did it again; or
      * silently loaded extensions built for the other stack's torch, which is
        undefined behaviour (see the WinError 127 note in isolation/metadata.py).

    Putting the ABI in the directory name keeps the sharing this root exists
    for -- two installs on the SAME stack still share one env -- while making
    incompatible reuse structurally impossible.
    """
    global _ABI_TAG
    if _ABI_TAG is not None:
        return _ABI_TAG

    from ..detection.backend import detect_backend
    from ..detection.cuda import (
        get_bootstrap_python_version,
        get_bootstrap_torch_version,
    )

    py = get_bootstrap_python_version() or "unknown"
    parts = ["py" + py.replace(".", "")]

    torch_v = get_bootstrap_torch_version()
    if torch_v:
        # major.minor only: torch patch releases do not break the C++ ABI, and
        # keying on them would rebuild multi-GB envs for nothing.
        parts.append("torch" + ".".join(torch_v.split(".")[:2]))
        backend, ver = detect_backend()
        if backend == "cuda" and ver:
            parts.append("cu" + ver.replace(".", ""))
        elif backend == "rocm" and ver:
            parts.append("rocm" + ver.replace(".", ""))
        else:
            parts.append(backend)
    else:
        # No torch in the bootstrap: the comfyui feature stays torch-less, so
        # there is no ABI to pin beyond the interpreter.
        parts.append("notorch")

    _ABI_TAG = _sanitize_pixi_name("-".join(parts))
    return _ABI_TAG


def _env_dir_name(env_name: str) -> str:
    """On-disk directory for a logical env name, ABI-qualified.

    ``env_name`` stays the logical identity (manifests, logs, config lookup);
    only the directory carries the ABI tag.
    """
    return f"{env_name}-{_abi_tag()}"


def _short_global_root():
    """Resolve workspace root. Defaults to %LOCALAPPDATA%\\Programs\\comfy-env
    on Windows (sits next to the ComfyUI Desktop install at
    %LOCALAPPDATA%\\Programs\\ComfyUI) so fresh installs never need admin --
    the old default `C:\\ce` required admin to create at drive root.
    Override via COMFY_ENV_ROOT.
    """
    global _ANNOUNCED_WS

    override = os.environ.get("COMFY_ENV_ROOT")
    if override:
        root = Path(override)
    elif sys.platform == "win32":
        root = _windows_local_appdata() / "Programs" / "comfy-env"
    else:
        root = Path.home() / ".ce"

    root.mkdir(parents=True, exist_ok=True)

    if not _ANNOUNCED_WS:
        print(f"[comfy-env] Workspace: {root}", file=sys.stderr, flush=True)
        legacy = Path(r"C:\ce\.pixi\envs")
        if sys.platform == "win32" and root != Path(r"C:\ce") and legacy.is_dir() and any(legacy.iterdir()):
            print(
                "[comfy-env] Legacy workspace detected at C:\\ce. "
                "Workspace has moved to %LOCALAPPDATA%\\Programs\\comfy-env -- "
                "please reinstall the node ('python install.py') and then "
                "delete C:\\ce.",
                file=sys.stderr, flush=True,
            )
        _ANNOUNCED_WS = True

    return root


def get_env_name(plugin_dir, config_path):
    """Compute the pixi env name for a node's isolated environment.

    Format: ``<plugin>`` for root-level configs, ``<plugin>-<subdir>``
    otherwise. Strips ``ComfyUI[-_]`` prefix, lowercases, and collapses any
    char outside ``[a-z0-9-]`` to a single dash (so a folder like
    ``Foo._disabled`` produces ``foo-disabled``, not ``foo.disabled`` which
    pixi rejects).
    """
    plugin_dir, config_path = Path(plugin_dir), Path(config_path)

    plugin_part = plugin_dir.name
    for prefix in ("ComfyUI-", "ComfyUI_", "comfyui-", "comfyui_"):
        if plugin_part.startswith(prefix):
            plugin_part = plugin_part[len(prefix):]
            break
    name = _sanitize_pixi_name(plugin_part)

    config_parent = config_path.parent.resolve()
    plugin_resolved = plugin_dir.resolve()
    if config_parent != plugin_resolved:
        try:
            rel = config_parent.relative_to(plugin_resolved)
            suffix_raw = rel.parts[-1] if rel.parts else ""
        except ValueError:
            suffix_raw = config_parent.name
        suffix = _sanitize_pixi_name(suffix_raw)
        if suffix:
            name = f"{name}-{suffix}" if name else suffix

    return name


def get_workspace_dir(comfyui_dir=None):
    """Return the single global comfy-env pixi workspace root.

    Shared across every ComfyUI install on this machine — env names act as
    the global identifier (conda-style). `comfyui_dir` is accepted for
    signature compatibility but ignored.
    """
    return _short_global_root()


_ORPHAN_WARNED = set()


def _warn_if_orphaned(envs_root: Path, env_name: str, target: Path):
    """One-shot notice when an ABI-unqualified env from before this change exists.

    Without it the only symptom of the rename is a multi-GB re-materialization
    with no explanation.
    """
    if env_name in _ORPHAN_WARNED:
        return
    _ORPHAN_WARNED.add(env_name)
    legacy = envs_root / env_name
    if legacy.is_dir() and not target.is_dir():
        print(
            f"[comfy-env] Env '{env_name}' exists unqualified at {legacy} but is "
            f"not ABI-tagged; it was built for an unknown python/torch/backend "
            f"and cannot be trusted for this stack ({_abi_tag()}). Materializing "
            f"{target.name} instead -- delete the old dir once you are happy.",
            file=sys.stderr, flush=True,
        )


def get_env_manifest_dir(env_name: str, comfyui_dir=None) -> Path:
    """Directory containing one env's `pixi.toml` (new per-env layout).

    `<workspace>/envs/<env_name>-<abi_tag>/`
    """
    envs_root = get_workspace_dir(comfyui_dir) / "envs"
    target = envs_root / _env_dir_name(env_name)
    _warn_if_orphaned(envs_root, env_name, target)
    return target




def resolve_pixi_manifest(env_root: Path) -> tuple[Path, str]:
    """Given a materialized env directory, return ``(manifest_path, env_pixi_name)``.

    Per-env layout only. ``env_root`` is always
    ``<workspace>/envs/<env_name>/.pixi/envs/default`` -- the per-env pixi
    manifest lives at ``<workspace>/envs/<env_name>/pixi.toml`` and the
    pixi environment inside that manifest is always named ``default``.
    """
    env_root = Path(env_root)
    # <ws>/envs/<name>/.pixi/envs/default -> walk up 3 to reach <ws>/envs/<name>/
    manifest_dir = env_root.parent.parent.parent
    manifest = manifest_dir / "pixi.toml"
    return (manifest, "default")


def get_workspace_env_dir(comfyui_dir, env_name):
    """Path to one environment's materialized site-packages root.

    Always ``<workspace>/envs/<env_name>-<abi_tag>/.pixi/envs/default/``.
    The ABI tag is what stops two ComfyUI installs on different
    python/torch/backend stacks from sharing an env that only one of them can
    actually load -- see _abi_tag(). No legacy fallback.
    """
    return get_env_manifest_dir(env_name, comfyui_dir) / ".pixi" / "envs" / "default"


_STAMP_FILE = "env.stamp.json"


def write_env_stamp(env_manifest_dir, torch_pin=None, provenance="unknown", log=None):
    """Record what an env was built from and against, next to its manifest.

    Written only after a successful install. `validate_env_stamp` checks it at
    bind time -- without this, an env is trusted purely because its directory
    exists, and a foreign-stack env gets loaded into torch's private
    multiprocessing ABI (reduce_storage/rebuild_cuda_tensor) which has no
    version handshake of its own.
    """
    import hashlib as _hashlib
    import json as _json

    from .. import __version__ as ce_version

    env_manifest_dir = Path(env_manifest_dir)
    lock = env_manifest_dir / "pixi.lock"
    lock_sha = None
    try:
        if lock.is_file():
            lock_sha = _hashlib.sha256(lock.read_bytes()).hexdigest()
    except OSError:
        pass
    stamp = {
        "comfy_env_version": ce_version,
        "abi_tag": _abi_tag(),
        "torch_pin": torch_pin,
        "provenance": provenance,
        "pixi_lock_sha256": lock_sha,
    }
    try:
        (env_manifest_dir / _STAMP_FILE).write_text(
            _json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
        if log:
            log(f"[comfy-env] Stamped {env_manifest_dir.name}: "
                f"abi={stamp['abi_tag']} torch={torch_pin} ({provenance})")
    except OSError as e:
        if log:
            log(f"[comfy-env] WARNING: could not write env stamp: {e}")


def validate_env_stamp(env_manifest_dir):
    """Check a materialized env's stamp against the current stack.

    Returns (ok, reason). Unstamped envs pass with a note -- they predate
    stamping, and hard-failing them would orphan every existing env (the
    don't-break-userspace case). A PRESENT stamp that disagrees on the ABI tag
    fails: that env was demonstrably built for a different stack, and binding
    to it is the silent-mismatch bug.
    """
    import json as _json

    p = Path(env_manifest_dir) / _STAMP_FILE
    if not p.is_file():
        return True, "unstamped (pre-stamping env; not verified)"
    try:
        stamp = _json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return True, f"stamp unreadable ({e}); not verified"
    want = _abi_tag()
    got = stamp.get("abi_tag")
    if got and got != want:
        return False, (
            f"built for abi={got}, current stack is abi={want} "
            f"(provenance={stamp.get('provenance')}, "
            f"torch_pin={stamp.get('torch_pin')})"
        )
    return True, f"abi={got} verified"


def _candidate_config_dirs():
    """Yield candidate ComfyUI Desktop config dirs, ordered most-specific
    first. On Windows, robust against SYSTEM-context APPDATA inherited from
    agent harnesses, scheduled tasks, or service-spawned shells: when
    APPDATA points at the systemprofile subtree (where ComfyUI never
    writes), we fall through to USERPROFILE-, USERNAME-, then a glob across
    C:\\Users\\* picking any user that has the userData dir."""
    if sys.platform == "darwin":
        return [Path.home() / "Library" / "Application Support" / "ComfyUI"]
    if sys.platform != "win32":
        return [Path.home() / ".config" / "ComfyUI"]

    seen = set()
    out = []
    def add(p):
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    appdata = os.environ.get("APPDATA", "")
    if appdata and "systemprofile" not in appdata.lower():
        add(Path(appdata) / "ComfyUI")
    userprofile = os.environ.get("USERPROFILE", "")
    if userprofile and "systemprofile" not in userprofile.lower():
        add(Path(userprofile) / "AppData" / "Roaming" / "ComfyUI")
    username = os.environ.get("USERNAME", "")
    if username and username.upper() != "SYSTEM":
        add(Path("C:/Users") / username / "AppData" / "Roaming" / "ComfyUI")
    try:
        from glob import glob as _glob
        skip = ("default", "default user", "public", "all users")
        for p in _glob(r"C:\Users\*\AppData\Roaming\ComfyUI"):
            parts = Path(p).parts
            user_seg = parts[2].lower() if len(parts) > 2 else ""
            if user_seg in skip:
                continue
            add(Path(p))
    except Exception:
        pass
    return out


def _find_desktop_source_dir():
    """Find ComfyUI source dir from the Desktop app's extra_models_config.yaml.

    The Electron desktop app stores config in its userData dir:
      macOS:   ~/Library/Application Support/ComfyUI/
      Windows: %APPDATA%/ComfyUI/
      Linux:   ~/.config/ComfyUI/

    The file extra_models_config.yaml contains a desktop_extensions section
    with a custom_nodes path pointing into the app bundle. The parent of
    that path is the ComfyUI source dir (where main.py, comfy/, requirements.txt live).
    """
    for config_dir in _candidate_config_dirs():
        yaml_path = config_dir / "extra_models_config.yaml"
        if not yaml_path.exists():
            continue
        try:
            content = yaml_path.read_text(encoding="utf-8")
            # Look for desktop_extensions.custom_nodes line
            in_desktop = False
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("desktop_extensions"):
                    in_desktop = True
                    continue
                if in_desktop and stripped.startswith("custom_nodes:"):
                    path_str = stripped.split(":", 1)[1].strip()
                    candidate = Path(path_str).parent
                    if (candidate / "main.py").exists():
                        return candidate
                if (not line.startswith(" ")
                        and not line.startswith("\t")
                        and ":" in line):
                    in_desktop = False
        except Exception:
            continue
    return None


def find_comfyui_dir_from_node(node_dir=None):
    """Find the ComfyUI user data directory (where .ce/, custom_nodes/, user/ live).

    This is where the workspace and pixi envs are created.
    On standard installs this is the same as the source dir.
    On Desktop app this is ~/Documents/ComfyUI/ (NOT the app bundle).
    """
    # Running inside ComfyUI server — base_path is the user data dir
    try:
        import folder_paths
        return Path(folder_paths.base_path)
    except ImportError:
        pass

    # Walk up from node_dir. abspath, NOT resolve(): a pack living behind a
    # junction/symlink (custom_nodes/Pack -> elsewhere/Pack) must walk up
    # through custom_nodes/ into the ComfyUI tree. resolve() follows the link
    # to the physical location, where no ComfyUI root exists, and the walk
    # returns None -- the out-of-process twin of the #8 identity bug (in-server
    # callers never got here because the folder_paths import short-circuits).
    if node_dir is not None:
        current = Path(os.path.abspath(node_dir))
        for _ in range(10):
            # Standard: has main.py + comfy/ (source dir IS the data dir)
            if (current / "main.py").exists() and (current / "comfy").exists():
                return current
            # Desktop app: has custom_nodes/ + user/ but no main.py
            if (current / "custom_nodes").is_dir() and (current / "user").is_dir():
                return current
            if current.parent == current:
                break
            current = current.parent
    return None


def find_comfyui_source_dir(node_dir=None):
    """Find the ComfyUI source directory (where main.py, comfy/, requirements.txt live).

    On standard installs this is the same as the data dir.
    On Desktop app this is inside the app bundle (e.g. ComfyUI.app/.../ComfyUI/).
    """
    # Running inside ComfyUI server — folder_paths module is in the source dir
    try:
        import folder_paths
        return Path(folder_paths.__file__).parent
    except ImportError:
        pass

    # Walk up from node_dir — if we find main.py + comfy/, that's it.
    # abspath, not resolve(): see find_comfyui_dir_from_node above.
    if node_dir is not None:
        current = Path(os.path.abspath(node_dir))
        for _ in range(10):
            if (current / "main.py").exists() and (current / "comfy").exists():
                return current
            if current.parent == current:
                break
            current = current.parent

    # Desktop app: source dir is in the app bundle, read from config
    return _find_desktop_source_dir()


_FOLDER_PATH_ATTR = {
    "input":  "get_input_directory",
    "output": "get_output_directory",
    "temp":   "get_temp_directory",
    "user":   "get_user_directory",
}


def _resolve_dst_via_folder_paths(dst):
    """If `dst` is a relative str whose first path segment names a known
    ComfyUI directory type, resolve that segment via ComfyUI's
    `folder_paths` module. Returns an absolute Path on success, or None
    if the smart resolution doesn't apply (caller then does its normal
    resolution).

    This is the single fix that makes writers of ComfyUI input/output/
    temp/user dirs match ComfyUI's runtime resolvers on every deployment
    shape — vanilla, Comfy Desktop with `inputDir` override, launches
    with `--input-directory` / `--base-directory` / etc.
    """
    if not isinstance(dst, str):
        return None
    if Path(dst).is_absolute():
        return None
    parts = dst.replace("\\", "/").split("/", 1)
    head = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    if head not in _FOLDER_PATH_ATTR:
        return None
    try:
        import folder_paths
        base = Path(getattr(folder_paths, _FOLDER_PATH_ATTR[head])())
        return base / rest if rest else base
    except Exception:
        # folder_paths not importable (e.g. helper called outside a
        # ComfyUI process). Fall through so caller-relative resolution
        # is attempted instead.
        return None


def copy_files(src, dst, pattern="*", overwrite=False):
    """Copy files matching `pattern` from `src` to `dst`.

    Both `src` and `dst` accept:
      - absolute Path/str: used as-is (back-compat)
      - relative Path/str: resolved against the CALLING SCRIPT's
        directory (via frame inspection). So `copy_files("assets",
        "input/3d", "**/*")` from inside a custom node's
        `prestartup_script.py` finds `<node>/assets/` and copies to the
        ComfyUI-configured input dir + `/3d`.

    `dst` special-case: a relative str whose first path segment is
    `input`, `output`, `temp`, or `user` — that segment is resolved via
    `folder_paths` (the single source of truth ComfyUI itself uses),
    guaranteeing writer/reader parity on hosts that redirect these
    paths (Comfy Desktop's inputDir override, `--input-directory`,
    `extra_model_paths.yaml`, etc.). Examples:
      - `"input"`                    → folder_paths.get_input_directory()
      - `"input/cad"`                → get_input_directory() / "cad"
      - `"user/default/workflows"`   → get_user_directory() / "default/workflows"
    """
    # Resolve the caller's directory HERE, in the public function body, where
    # depth 1 means "whoever called copy_files" -- a fact about the call
    # contract. This used to live inside the nested helper below at depth 2,
    # which encoded an INTERNAL structural fact (that a helper exists between
    # the caller and the frame walk). Adding any frame -- a decorator, another
    # helper, a deprecation shim -- shifted it silently: `__file__` on the
    # wrong frame is still a valid path, so `src` resolved under comfy-env's
    # own directory, `src.exists()` returned False, and the function returned
    # 0 having copied nothing. No exception, no log, no thread to pull.
    _frame_file = sys._getframe(1).f_globals.get("__file__")
    caller_dir = (  # abspath, not resolve(): keep symlink spelling (#8)
        Path(os.path.abspath(_frame_file)).parent if _frame_file else Path.cwd()
    )

    def _resolve_relative(p):
        pp = Path(p)
        return pp if pp.is_absolute() else caller_dir / pp

    dst_smart = _resolve_dst_via_folder_paths(dst)
    dst = dst_smart if dst_smart is not None else _resolve_relative(dst)
    src = _resolve_relative(src)

    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in src.glob(pattern):
        if f.is_file():
            target = dst / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not target.exists():
                shutil.copy2(f, target)
                copied += 1
    return copied
