"""Contract: a preview handed to ComfyUI's ProgressBar in a worker reaches the
host's PROGRESS_BAR_HOOK as upstream's PreviewImageTuple.

Upstream's contract (comfy/utils.py, comfy_execution/progress.py): the bar
calls hook(value, total, preview, node_id=...), preview is
(format, PIL.Image, max_size), and the host consumer does the downscale
itself (ImageOps.contain in send_image), so the worker must hand over the
image at its original size. The send_sync preview path had an end-to-end
test; this path (ProgressBar -> worker hook -> report_progress ->
_handle_progress -> PROGRESS_BAR_HOOK) did not.

Needs a ComfyUI checkout: set COMFYUI_DIR. Skipped otherwise.
"""

import os
import sys
from pathlib import Path

import pytest

from comfy_env.isolation.pool import _handle_progress
from comfy_env.isolation.workers.subprocess import SubprocessWorker

pytestmark = pytest.mark.comfyui

COMFYUI_DIR = os.environ.get("COMFYUI_DIR")
if not COMFYUI_DIR:
    pytest.skip("COMFYUI_DIR not set", allow_module_level=True)

FIXTURES = Path(__file__).parent / "fixtures"

# comfy.model_management parses ComfyUI's args on first import and, on a
# CPU-only torch, asserts unless --cpu was given. That happens on the FIRST
# comfy import in the pytest process, whichever test module triggers it, so
# every comfyui-marked module that imports comfy does this before anything.
def _parse_cpu_args():
    sys.path.insert(0, COMFYUI_DIR)
    import comfy.options
    comfy.options.enable_args_parsing()
    if "--cpu" not in sys.argv:
        sys.argv = [sys.argv[0], "--cpu"]
    import comfy.cli_args  # noqa: F401  -- parses now, under --cpu
    sys.path.remove(COMFYUI_DIR)


_parse_cpu_args()


@pytest.fixture()
def host_hook():
    """The host side of ComfyUI, enough for _handle_progress to run, and a
    capturing PROGRESS_BAR_HOOK."""
    sys.path.insert(0, COMFYUI_DIR)
    try:
        import comfy.model_management  # noqa: F401  -- the import _handle_progress guards
        import comfy.utils
        seen = []
        comfy.utils.set_progress_bar_global_hook(lambda v, t, p, node_id=None: seen.append((v, t, p)))
        yield seen
    finally:
        comfy.utils.set_progress_bar_global_hook(None)
        sys.path.remove(COMFYUI_DIR)


def test_upstream_bar_signature_and_tuple_shape(host_hook):
    """Canary on the contract the worker's hook signature relies on."""
    import inspect
    import comfy.utils
    from comfy_execution import progress
    src = inspect.getsource(comfy.utils.ProgressBar.update_absolute)
    assert "node_id=" in src, "ProgressBar no longer passes node_id= to the hook"
    assert hasattr(progress, "PreviewImageTuple")


def test_preview_reaches_the_host_hook_at_original_size(host_hook):
    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                         sys_path=[COMFYUI_DIR], name="preview-worker")
    w.register_callback("report_progress", _handle_progress)
    try:
        w.call_method(module_name="progress_preview_node", class_name="Previewer",
                      method_name="run", kwargs={"steps": 3}, timeout=120.0)
    finally:
        w.shutdown()
    assert [v for v, t, p in host_hook] == [1, 2, 3]
    fmt, img, max_size = host_hook[-1][2]
    assert fmt == "JPEG" and max_size == 512
    assert img.size == (40, 30), "the worker must not downscale; the host does"
