"""
Import stub system for isolated node packs.

This module provides automatic import stubbing for packages that exist only
in the isolated pixi environment, not in the host ComfyUI Python.

How it works:
1. Scan pixi environment's site-packages for installed packages
2. Look up import names from top_level.txt in .dist-info directories
3. Inject stub modules directly into sys.modules for missing packages
4. Stubs allow class definitions to parse without the real packages
5. Real packages are used when FUNCTION runs in the isolated worker

Usage:
    # In node pack's __init__.py, BEFORE importing nodes:
    from comfy_env import setup_isolated_imports
    setup_isolated_imports(__file__)

    from .nodes import NODE_CLASS_MAPPINGS  # Now works!
"""

import sys
import types
from pathlib import Path
from typing import List, Set


def _log(msg: str) -> None:
    """Log with immediate flush to stderr (visible on Windows subprocess)."""
    print(msg, file=sys.stderr, flush=True)


class _StubModule(types.ModuleType):
    """A stub module that accepts any attribute access or call."""

    def __init__(self, name: str):
        super().__init__(name)
        self.__path__ = []  # Make it a package
        self.__file__ = f"<stub:{name}>"
        self._stub_name = name

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(name)
        return _StubObject(f"{self._stub_name}.{name}")

    def __repr__(self):
        return f"<StubModule '{self._stub_name}'>"


class _StubObject:
    """A stub object that accepts any operation."""

    def __init__(self, name: str = "stub"):
        self._stub_name = name

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(name)
        return _StubObject(f"{self._stub_name}.{name}")

    def __call__(self, *args, **kwargs):
        return _StubObject(f"{self._stub_name}()")

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    def __bool__(self):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __repr__(self):
        return f"<Stub '{self._stub_name}'>"

    def __add__(self, other): return self
    def __radd__(self, other): return self
    def __sub__(self, other): return self
    def __rsub__(self, other): return self
    def __mul__(self, other): return self
    def __rmul__(self, other): return self
    def __truediv__(self, other): return self
    def __rtruediv__(self, other): return self
    def __eq__(self, other): return False
    def __ne__(self, other): return True
    def __lt__(self, other): return False
    def __le__(self, other): return False
    def __gt__(self, other): return False
    def __ge__(self, other): return False
    def __hash__(self): return hash(self._stub_name)
    def __getitem__(self, key): return _StubObject(f"{self._stub_name}[{key}]")
    def __setitem__(self, key, value): pass
    def __contains__(self, item): return False


def _get_import_names_from_pixi(node_dir: Path) -> Set[str]:
    """
    Get import names from pixi environment using top_level.txt metadata.

    This properly maps package names to import names (e.g., libigl -> igl,
    PyYAML -> yaml) by reading the canonical top_level.txt files.

    Returns:
        Set of import names that should be stubbed.
    """
    import_names = set()

    pixi_base = node_dir / ".pixi" / "envs" / "default"

    # Find site-packages (different paths on Windows vs Linux)
    site_packages = None
    win_site = pixi_base / "Lib" / "site-packages"
    if win_site.exists():
        site_packages = win_site
    else:
        pixi_lib = pixi_base / "lib"
        if pixi_lib.exists():
            python_dirs = list(pixi_lib.glob("python3.*"))
            if python_dirs:
                site_packages = python_dirs[0] / "site-packages"

    if site_packages is None or not site_packages.exists():
        return import_names

    _log(f"[comfy-env] Scanning: {site_packages}")

    # PRIMARY: Read top_level.txt from all .dist-info directories
    for dist_info in site_packages.glob("*.dist-info"):
        top_level_file = dist_info / "top_level.txt"
        if top_level_file.exists():
            try:
                for line in top_level_file.read_text(encoding="utf-8").splitlines():
                    name = line.strip()
                    if name and not name.startswith('#'):
                        # Extract just the top-level name
                        top_name = name.replace('\\', '/').split('/')[0]
                        if top_name:
                            import_names.add(top_name)
            except Exception:
                pass

    # FALLBACK: Scan for packages/modules not covered by dist-info
    for item in site_packages.iterdir():
        name = item.name

        if name.startswith('_') or name.startswith('.'):
            continue
        if name.endswith('.dist-info') or name.endswith('.egg-info'):
            continue
        if name in {'bin', 'share', 'include', 'etc'}:
            continue

        # Package directory (has __init__.py)
        if item.is_dir() and (item / "__init__.py").exists():
            import_names.add(name)
            continue

        # Namespace package (directory without __init__.py but has submodules)
        if item.is_dir():
            has_py = any(item.glob("*.py"))
            has_subpkg = any((item / d / "__init__.py").exists() for d in item.iterdir() if d.is_dir())
            if has_py or has_subpkg:
                import_names.add(name)
            continue

        # Single-file module (.py)
        if name.endswith('.py'):
            import_names.add(name[:-3])
            continue

        # Extension module (.so on Linux, .pyd on Windows)
        if name.endswith('.so') or name.endswith('.pyd'):
            module_name = name.split('.')[0]
            import_names.add(module_name)

    return import_names


def _filter_to_missing(import_names: Set[str]) -> Set[str]:
    """Filter to only imports not available in host Python."""
    missing = set()

    for name in import_names:
        # Skip if already in sys.modules
        if name in sys.modules:
            continue

        # Try to import
        try:
            __import__(name)
        except ImportError:
            missing.add(name)
        except Exception:
            # Other errors (DLL load, etc.) - stub these too
            missing.add(name)

    return missing


# Track what we stubbed for cleanup
_stubbed_modules: Set[str] = set()


def setup_isolated_imports(init_file: str) -> List[str]:
    """
    Set up import stubs for packages in the pixi environment but not in host Python.

    Call this BEFORE importing your nodes module.

    Args:
        init_file: The __file__ of the calling module (usually __file__ from __init__.py)

    Returns:
        List of import names that were stubbed.

    Example:
        from comfy_env import setup_isolated_imports
        setup_isolated_imports(__file__)

        from .nodes import NODE_CLASS_MAPPINGS  # Now works!
    """
    global _stubbed_modules

    node_dir = Path(init_file).resolve().parent

    # Get all import names from pixi environment
    pixi_imports = _get_import_names_from_pixi(node_dir)

    if not pixi_imports:
        _log("[comfy-env] No pixi environment found")
        return []

    # Filter to only those missing in host
    missing = _filter_to_missing(pixi_imports)

    if not missing:
        _log("[comfy-env] All packages available in host")
        return []

    # Direct injection into sys.modules - simple and reliable
    for name in missing:
        if name not in sys.modules:
            sys.modules[name] = _StubModule(name)
            _stubbed_modules.add(name)

    stubbed = sorted(_stubbed_modules)
    if len(stubbed) <= 10:
        _log(f"[comfy-env] Injected {len(stubbed)} stubs: {', '.join(stubbed)}")
    else:
        _log(f"[comfy-env] Injected {len(stubbed)} stubs: {', '.join(stubbed[:10])}... +{len(stubbed)-10} more")

    return stubbed


def cleanup_stubs():
    """Remove injected stub modules from sys.modules."""
    global _stubbed_modules

    for name in list(_stubbed_modules):
        if name in sys.modules and isinstance(sys.modules[name], _StubModule):
            del sys.modules[name]

    _stubbed_modules.clear()
