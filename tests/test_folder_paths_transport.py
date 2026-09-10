"""Contract: the host's folder_paths state reaches the worker intact.

A pack's nodes resolve models through `folder_paths`, so an isolated worker
must see the SAME registry the host does -- every category, every directory,
in order, with the extension sets rebuilt. The parent snapshots that state at
spawn (`subprocess.py:631-657`) and the worker applies it before any pack code
runs (`_persistent_worker.py:939-968`).

Nothing tested this. Two properties in particular were unguarded:

* **The set round trip.** JSON has no sets, so the parent serializes
  extensions as `sorted(exts)` and the worker rebuilds `set(...)`. If either
  side drifted, `get_filename_list` would return nothing in workers only --
  a combo silently going empty, with no error anywhere.
* **The silent fallback.** The parent's snapshot sits under a bare
  `except Exception: pass`. If it ever throws, the worker keeps its own
  defaults and resolves models against the wrong root, saying nothing.

These run the REAL worker in a subprocess with the current interpreter, and
stub `folder_paths` on both sides so no ComfyUI checkout is needed.
"""

import importlib.util
import sys
import textwrap

import pytest

from comfy_env.isolation.workers.subprocess import SubprocessWorker

# A registry shaped like the real one: a plain category, a two-directory
# category (how upstream absorbs a rename -- unet -> diffusion_models), a
# non-default extension set, and an empty set meaning "accept everything".
REGISTRY = {
    "checkpoints":      ([r"/models/checkpoints"], {".safetensors", ".ckpt"}),
    "diffusion_models": ([r"/models/unet", r"/models/diffusion_models"], {".safetensors"}),
    "configs":          ([r"/models/configs"], {".yaml"}),
    "custom_nodes":     ([r"/custom_nodes"], set()),
}
DIRS = {
    "base_path":        r"/comfy-base",
    "input_directory":  r"/comfy-base/input",
    "output_directory": r"/elsewhere/output",   # deliberately not under base
    "temp_directory":   r"/comfy-base/temp",
    "user_directory":   r"/comfy-base/user",
}

_STUB = textwrap.dedent('''
    """Minimal stand-in for ComfyUI's folder_paths, host and worker side."""
    folder_names_and_paths = {}
    base_path = None
    input_directory = output_directory = temp_directory = user_directory = None

    def set_input_directory(d):
        global input_directory; input_directory = d
    def set_output_directory(d):
        global output_directory; output_directory = d
    def set_temp_directory(d):
        global temp_directory; temp_directory = d
    def set_user_directory(d):
        global user_directory; user_directory = d
    def get_input_directory():  return input_directory
    def get_output_directory(): return output_directory
    def get_temp_directory():   return temp_directory
    def get_user_directory():   return user_directory
''')

_PROBE = textwrap.dedent('''
    """Reports what folder_paths looks like INSIDE the worker."""
    def snapshot():
        import folder_paths as fp
        return {
            "base_path":        fp.base_path,
            "input_directory":  fp.get_input_directory(),
            "output_directory": fp.get_output_directory(),
            "temp_directory":   fp.get_temp_directory(),
            "user_directory":   fp.get_user_directory(),
            # sorted() so the comparison is stable; the TYPE is asserted
            # separately, because that is the half JSON can silently lose.
            "registry":  {k: [list(v[0]), sorted(v[1])]
                          for k, v in fp.folder_names_and_paths.items()},
            "ext_types": {k: type(v[1]).__name__
                          for k, v in fp.folder_names_and_paths.items()},
        }
''')


def _load_stub(path):
    """Load the stub as a module object without touching sys.path."""
    spec = importlib.util.spec_from_file_location("folder_paths", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def seen(tmp_path, monkeypatch):
    """Spawn a real worker with a stubbed host folder_paths; return what it saw."""
    (tmp_path / "folder_paths.py").write_text(_STUB)
    (tmp_path / "fp_probe.py").write_text(_PROBE)

    # Host side: the snapshot does `import folder_paths`, so bind the stub into
    # sys.modules and populate it with the state under test.
    #
    # setitem, NOT syspath_prepend + import: `find_comfyui_source_dir` also does
    # `import folder_paths` and reads `base_path` off it (environment/cache.py:648).
    # A stub left behind in sys.modules therefore silently redirects every later
    # ComfyUI-root lookup in the session -- which is exactly what an earlier draft
    # of this test did, breaking four tests in another file. setitem records that
    # the key was absent and deletes it on teardown; a plain import does not.
    host_fp = _load_stub(tmp_path / "folder_paths.py")
    host_fp.folder_names_and_paths = REGISTRY
    host_fp.base_path = DIRS["base_path"]
    host_fp.set_input_directory(DIRS["input_directory"])
    host_fp.set_output_directory(DIRS["output_directory"])
    host_fp.set_temp_directory(DIRS["temp_directory"])
    host_fp.set_user_directory(DIRS["user_directory"])
    monkeypatch.setitem(sys.modules, "folder_paths", host_fp)

    worker = SubprocessWorker(
        python=sys.executable,
        working_dir=tmp_path,
        name="folder-paths-transport",
    )
    try:
        yield worker.call_module(module="fp_probe", func="snapshot")
    finally:
        worker.shutdown()


def test_every_category_crosses(seen):
    assert set(seen["registry"]) == set(REGISTRY)


@pytest.mark.parametrize("category", sorted(REGISTRY))
def test_directories_cross_in_order(seen, category):
    # Order is load bearing: the FIRST directory is what "where do I write
    # this" logic picks, and it is what is_default=True buys.
    assert seen["registry"][category][0] == REGISTRY[category][0]


@pytest.mark.parametrize("category", sorted(REGISTRY))
def test_extensions_survive_the_json_set_collapse(seen, category):
    assert seen["registry"][category][1] == sorted(REGISTRY[category][1])
    # The type matters as much as the contents: folder_paths' own
    # filter_files_extensions does a membership test, and a list that should
    # have been a set still passes it -- until someone relies on set algebra.
    assert seen["ext_types"][category] == "set"


def test_two_directory_category_keeps_both(seen):
    # unet -> diffusion_models is a rename absorbed by listing both folders.
    # Dropping either one silently orphans models already on disk.
    assert len(seen["registry"]["diffusion_models"][0]) == 2


def test_empty_extension_set_is_not_confused_with_missing(seen):
    # An empty set means "accept every file", not "accept none". If the
    # transport turned it into a missing key the category would go dark.
    assert "custom_nodes" in seen["registry"]
    assert seen["registry"]["custom_nodes"][1] == []
    assert seen["ext_types"]["custom_nodes"] == "set"


@pytest.mark.parametrize("key", sorted(DIRS))
def test_working_directories_cross(seen, key):
    assert seen[key] == DIRS[key]
