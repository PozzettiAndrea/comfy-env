"""Two contracts a worker used to break silently.

* logging.info() from pack code reaches the host. The worker attached a
  handler to logging.root but never lowered the LOGGER's level, and a handler
  never sees a record the logger filtered first -- so INFO vanished while
  WARNING kept working, which is exactly why nobody noticed.
* folder_paths.models_dir in the worker is the HOST's. The registry crossed;
  the scalar it was built from did not, so a pack carving out its own model
  subfolder pointed at the wrong root on any non-default layout.
"""

import logging
import sys
from pathlib import Path

import pytest

from comfy_env.isolation.workers.subprocess import SubprocessWorker

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def worker(monkeypatch):
    # Pretend to be a host whose root logger admits INFO, as ComfyUI's
    # setup_logger does (15 = DETAIL by default; INFO is the conservative bound).
    monkeypatch.setattr(logging.getLogger(), "level", logging.INFO)
    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                         name="loglevel-test-worker")
    yield w
    w.shutdown()


def test_logging_info_from_pack_code_reaches_the_host(worker, capsys):
    worker.call_module(module="log_and_paths_node", func="emit_at_every_level")
    err = capsys.readouterr().err
    assert "INFO-LINE" in err, "logging.info() was filtered before the handler"
    assert "WARNING-LINE" in err
    # DEBUG is below the host's level and must stay out -- the point is
    # parity with the host, not "forward everything".
    assert "DEBUG-LINE" not in err


def test_models_dir_is_the_hosts(worker, monkeypatch, tmp_path):
    """The parent snapshots folder_paths at spawn; make the host's models_dir
    unmistakable and check the worker reports it back."""
    # A minimal folder_paths.py in a fake ComfyUI base. The parent imports
    # it to snapshot; the worker imports the SAME file off disk (comfy-env
    # puts the base on its sys.path), with the worker's own stale default
    # models_dir -- which is the bug: only the snapshot can correct it.
    (tmp_path / "folder_paths.py").write_text(
        "import os\n"
        "base_path = os.path.dirname(os.path.realpath(__file__))\n"
        "models_dir = os.path.join(base_path, 'models')   # the DEFAULT, stale in a worker\n"
        "folder_names_and_paths = {}\n"
        "def get_input_directory():  return base_path + '/input'\n"
        "def get_output_directory(): return base_path + '/output'\n"
        "def get_temp_directory():   return base_path + '/temp'\n"
        "def get_user_directory():   return base_path + '/user'\n"
        "def set_input_directory(p): pass\n"
        "def set_output_directory(p): pass\n"
        "def set_temp_directory(p): pass\n"
        "def set_user_directory(p): pass\n"
    )
    import importlib
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("folder_paths", None)
    fake_fp = importlib.import_module("folder_paths")
    # The host was launched with --models-directory: point somewhere the
    # worker's default cannot coincide with.
    fake_fp.models_dir = str(tmp_path / "somewhere-else" / "models")

    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                         name="models-dir-test-worker")
    try:
        got = w.call_module(module="log_and_paths_node", func="read_models_dir")
    finally:
        w.shutdown()
    assert got == fake_fp.models_dir
