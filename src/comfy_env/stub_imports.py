"""
Import stub system for isolated node packs.

This module provides automatic import stubbing for packages that exist only
in the isolated pixi environment, not in the host ComfyUI Python.

How it works:
1. Read package names from comfy-env.toml
2. Look up their import names from top_level.txt in the pixi environment
3. Register import hooks that provide stub modules for those imports
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
from typing import Dict, List, Optional, Set


class _StubModule(types.ModuleType):
    """
    A stub module that accepts any attribute access or call.
    """

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
    """
    A stub object that accepts any operation.
    """

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


class _StubFinder:
    """Import hook finder that provides stub modules for specified packages."""

    def __init__(self, stub_packages: Set[str]):
        self.stub_packages = stub_packages

    def find_module(self, fullname: str, path=None):
        top_level = fullname.split('.')[0]
        if top_level in self.stub_packages:
            return _StubLoader(self.stub_packages)
        return None


class _StubLoader:
    """Import hook loader that creates stub modules."""

    def __init__(self, stub_packages: Set[str]):
        self.stub_packages = stub_packages

    def load_module(self, fullname: str):
        if fullname in sys.modules:
            return sys.modules[fullname]

        module = _StubModule(fullname)
        module.__loader__ = self

        if '.' in fullname:
            parent = fullname.rsplit('.', 1)[0]
            module.__package__ = parent
            if parent not in sys.modules:
                self.load_module(parent)
        else:
            module.__package__ = fullname

        sys.modules[fullname] = module
        return module


def _normalize_package_name(name: str) -> str:
    """Normalize package name for comparison (PEP 503)."""
    return name.lower().replace('-', '_').replace('.', '_')


def _get_import_names_from_pixi(node_dir: Path) -> Set[str]:
    """
    Get import names by scanning the pixi environment's site-packages.

    Finds all importable packages by looking for:
    1. Directories with __init__.py (packages)
    2. .py files (single-file modules)
    3. .so/.pyd files (extension modules)

    Returns:
        Set of import names that should be stubbed.
    """
    import_names = set()

    pixi_base = node_dir / ".pixi" / "envs" / "default"

    # Find site-packages (different paths on Windows vs Linux)
    # Linux: .pixi/envs/default/lib/python3.x/site-packages
    # Windows: .pixi/envs/default/Lib/site-packages
    site_packages = None

    # Try Windows path first (Lib/site-packages)
    win_site = pixi_base / "Lib" / "site-packages"
    if win_site.exists():
        site_packages = win_site
    else:
        # Try Linux path (lib/python3.x/site-packages)
        pixi_lib = pixi_base / "lib"
        if pixi_lib.exists():
            python_dirs = list(pixi_lib.glob("python3.*"))
            if python_dirs:
                site_packages = python_dirs[0] / "site-packages"

    if site_packages is None or not site_packages.exists():
        return import_names

    # Scan for importable modules
    for item in site_packages.iterdir():
        name = item.name

        # Skip private/internal items
        if name.startswith('_') or name.startswith('.'):
            continue

        # Skip dist-info and egg-info directories
        if name.endswith('.dist-info') or name.endswith('.egg-info'):
            continue

        # Skip common non-module items
        if name in {'bin', 'share', 'include', 'etc'}:
            continue

        # Package directory (has __init__.py)
        if item.is_dir():
            if (item / "__init__.py").exists():
                import_names.add(name)
            continue

        # Single-file module (.py)
        if name.endswith('.py'):
            import_names.add(name[:-3])
            continue

        # Extension module (.so on Linux, .pyd on Windows)
        if '.cpython-' in name and (name.endswith('.so') or name.endswith('.pyd')):
            # Extract module name: foo.cpython-311-x86_64-linux-gnu.so -> foo
            module_name = name.split('.')[0]
            import_names.add(module_name)
            continue

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
            # Other errors - don't stub, let real error surface
            pass

    return missing


# Track whether we've already set up stubs
_stub_finder: Optional[_StubFinder] = None


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
    global _stub_finder

    node_dir = Path(init_file).resolve().parent

    # Get all import names from pixi environment
    pixi_imports = _get_import_names_from_pixi(node_dir)

    if not pixi_imports:
        print("[comfy-env] No pixi environment found, skipping import stubbing")
        return []

    # Filter to only those missing in host
    missing = _filter_to_missing(pixi_imports)

    if not missing:
        print("[comfy-env] All pixi packages available in host, no stubbing needed")
        return []

    # Remove old finder if exists
    if _stub_finder is not None:
        try:
            sys.meta_path.remove(_stub_finder)
        except ValueError:
            pass

    # Register new finder
    _stub_finder = _StubFinder(missing)
    sys.meta_path.insert(0, _stub_finder)

    stubbed = sorted(missing)
    if len(stubbed) <= 10:
        print(f"[comfy-env] Stubbed {len(stubbed)} imports: {', '.join(stubbed)}")
    else:
        print(f"[comfy-env] Stubbed {len(stubbed)} imports: {', '.join(stubbed[:10])}... and {len(stubbed)-10} more")

    return stubbed


def cleanup_stubs():
    """Remove the stub import hooks."""
    global _stub_finder

    if _stub_finder is not None:
        try:
            sys.meta_path.remove(_stub_finder)
        except ValueError:
            pass

        # Remove stubbed modules from sys.modules
        to_remove = [
            name for name in sys.modules
            if isinstance(sys.modules[name], _StubModule)
        ]
        for name in to_remove:
            del sys.modules[name]

        _stub_finder = None
