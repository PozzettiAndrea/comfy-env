"""Contract: a preview handed to ComfyUI's ProgressBar in a worker reaches the
host's PROGRESS_BAR_HOOK as upstream's PreviewImageTuple, already fitted.

Upstream's contract (comfy/utils.py, comfy_execution/progress.py): the bar
calls hook(value, total, preview, node_id=...), preview is
(format, PIL.Image, max_size), and server.send_image fits the image into a
max_size box (ImageOps.contain) before it goes to the browser. The worker
does that same fit before the frame crosses, so a full-resolution preview
with a max_size costs a thumbnail on the socket, not a dropped frame; the
1 MiB cap on a forwarded frame only bites when max_size is None. The
send_sync preview path had an end-to-end test; this path (ProgressBar ->
worker hook -> report_progress -> _handle_progress -> PROGRESS_BAR_HOOK)
did not.

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


def _run(host_hook, cls, **kwargs):
    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                         sys_path=[COMFYUI_DIR], name="preview-worker")
    w.register_callback("report_progress", _handle_progress)
    try:
        w.call_method(module_name="progress_preview_node", class_name=cls,
                      method_name="run", kwargs=kwargs, timeout=120.0)
    finally:
        w.shutdown()
    return host_hook


def test_preview_reaches_the_host_hook_fitted_like_send_image(host_hook):
    """A 40x30 frame with max_size=512 arrives at 512x384: ImageOps.contain
    scales up as well as down, and the worker mirrors it."""
    seen = _run(host_hook, "Previewer", steps=3)
    assert [v for v, t, p in seen] == [1, 2, 3]
    fmt, img, max_size = seen[-1][2]
    assert fmt == "JPEG" and max_size == 512
    assert img.size == (512, 384), "the worker must fit the frame the way send_image does"


def test_a_huge_preview_with_a_max_size_is_a_thumbnail_not_a_dropped_frame(host_hook):
    """2048x2048 PNG noise is several MB at full size, over the forwarded
    cap; fitted to 512 it crosses and the browser gets its preview."""
    seen = _run(host_hook, "BigPreviewer", max_size=512, fmt="PNG")
    assert len(seen) == 1
    fmt, img, max_size = seen[-1][2]
    assert fmt == "PNG" and max_size == 512 and img.size == (512, 512)


def test_a_huge_preview_with_no_max_size_is_dropped_and_the_tick_still_goes(host_hook):
    """The one case the cap still guards: no fit requested, frame far over
    1 MiB. Natively this floods the browser; here the image is left out and
    the progress value still arrives."""
    seen = _run(host_hook, "BigPreviewer", max_size=0, fmt="PNG")
    assert [(v, t) for v, t, p in seen] == [(1, 1)]
    assert seen[-1][2] is None
