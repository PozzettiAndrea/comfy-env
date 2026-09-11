"""Contract: a pack that imports ComfyUI's `server` module loads and runs in
an isolated process, and its send_sync reaches the host's PromptServer.

154 of 493 surveyed packs do `from server import PromptServer` at module
level. The real server.py imports aiohttp at line 32, so in a lean env that
line failed before any node was defined, and the error read like a broken
ComfyUI install. Of those packs, 123 register HTTP routes, which run in the
host and are not forwarded (deliberately unsupported, and the stand-in says
so); the other 31 use send_sync, client_id and send_progress_text, which are
what the stand-in forwards.
"""

import sys
import types
from pathlib import Path

import pytest

import comfy_env.isolation.metadata as md
from comfy_env import server_stub
from comfy_env.isolation.pool import _handle_send_sync, _handle_send_progress_text
from comfy_env.isolation.workers import WorkerError
from comfy_env.isolation.workers.subprocess import SubprocessWorker

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------
# the stand-in itself
# --------------------------------------------------------------------------

class TestStandIn:
    def test_installs_the_names_packs_import(self):
        mods = {}
        server_stub.install(mods)
        mod = mods["server"]
        assert mod.PromptServer.instance is not None
        assert mod.BinaryEventTypes.UNENCODED_PREVIEW_IMAGE == 2
        assert mod.__comfy_env_stub__ is True

    def test_no_sender_is_a_silent_no_op(self):
        """The metadata scan installs the stand-in with nothing to talk to."""
        inst = server_stub.PromptServer.instance
        inst._sender = None
        inst.send_sync("e", {"a": 1})
        inst.send_progress_text("t", "1")

    def test_off_surface_names_raise_and_say_what_is_forwarded(self):
        inst = server_stub.PromptServer.instance
        with pytest.raises(AttributeError) as ei:
            inst.prompt_queue
        msg = str(ei.value)
        assert "not available in an isolated node process" in msg
        for name in server_stub.SURFACE:
            assert name in msg
        assert "routes" in msg
        # so a pack's `if hasattr(PromptServer.instance, "routes")` guard skips
        assert not hasattr(inst, "routes")

    def test_progress_text_bytes_become_str(self):
        inst = server_stub.PromptServer.instance
        seen = []
        inst._sender = lambda m, **p: seen.append((m, p))
        try:
            inst.send_progress_text(b"hi", 7)
        finally:
            inst._sender = None
        assert seen == [("send_progress_text", {"text": "hi", "node_id": "7", "sid": None})]


# --------------------------------------------------------------------------
# the host handlers
# --------------------------------------------------------------------------

class _FakeInstance:
    client_id = "host-client"

    def __init__(self):
        self.sent = []

    def send_sync(self, event, data, sid=None):
        self.sent.append(("send_sync", event, data, sid))

    def send_progress_text(self, text, node_id, sid=None):
        self.sent.append(("send_progress_text", text, node_id, sid))


@pytest.fixture()
def host_server(monkeypatch):
    mod = types.ModuleType("server")
    mod.PromptServer = type("PromptServer", (), {"instance": _FakeInstance()})
    monkeypatch.setitem(sys.modules, "server", mod)
    return mod.PromptServer.instance


class TestHostHandlers:
    def test_sid_none_stays_a_broadcast(self, host_server):
        _handle_send_sync({"event": "e", "data": {"a": 1}, "sid": None})
        assert host_server.sent == [("send_sync", "e", {"a": 1}, None)]

    def test_any_other_sid_means_the_hosts_client(self, host_server):
        """The worker's idea of a socket id is never a real one; the only
        client a pack can mean is the one that queued the prompt."""
        _handle_send_sync({"event": "e", "data": {}, "sid": "whatever-the-pack-had"})
        assert host_server.sent[0][3] == "host-client"

    def test_bytes_and_previews_are_rebuilt(self, host_server):
        import base64
        _handle_send_sync({"event": 1, "data": {"__b64__": base64.b64encode(b"\x00\x01").decode()}, "sid": None})
        assert host_server.sent[0][2] == b"\x00\x01"

    def test_progress_text_is_packed_by_the_host(self, host_server):
        _handle_send_progress_text({"text": "hi", "node_id": "7", "sid": "x"})
        assert host_server.sent == [("send_progress_text", "hi", "7", "host-client")]

    def test_outside_comfyui_is_a_no_op(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "server", raising=False)
        assert _handle_send_sync({"event": "e", "data": {}, "sid": None}) == {}


# --------------------------------------------------------------------------
# the metadata scan
# --------------------------------------------------------------------------

def test_scan_registers_a_pack_that_imports_server(tmp_path):
    """The whole point: before the stand-in, this pack's scan failed at its
    first line and registered nothing."""
    pkg = tmp_path / "server_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        (FIXTURES / "server_user_node.py").read_text(encoding="utf-8")
        + "\nNODE_CLASS_MAPPINGS = {'ServerUser': ServerUser}\nNODE_DISPLAY_NAME_MAPPINGS = {}\n",
        encoding="utf-8")
    env_dir = tmp_path / "env"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").symlink_to(sys.executable)
    payload = md.fetch_metadata(env_dir, "server_pkg", tmp_path)
    assert "ServerUser" in payload["nodes"], payload


# --------------------------------------------------------------------------
# the worker, end to end
# --------------------------------------------------------------------------

@pytest.fixture()
def worker(host_server):
    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                         name="server-stub-worker")
    frames = []
    w.register_callback("send_sync", lambda r: (frames.append(("send_sync", r)), _handle_send_sync(r))[1])
    w.register_callback("send_progress_text",
                        lambda r: (frames.append(("send_progress_text", r)), _handle_send_progress_text(r))[1])
    w.frames = frames
    yield w
    w.shutdown()


def _call(worker, method, **kw):
    return worker.call_method(module_name="server_user_node", class_name="ServerUser",
                              method_name=method, kwargs={"x": 3}, timeout=60.0, **kw)


def test_send_sync_reaches_the_hosts_prompt_server(worker, host_server):
    out = _call(worker, "run")
    events = [(e, d, sid) for (k, e, d, sid) in host_server.sent if k == "send_sync"]
    assert ("server_user.loaded", {"at": "import"}, None) in events, "import-time send_sync"
    assert ("server_user.dict", {"x": 3}, None) in events
    assert ("server_user.targeted", {"x": 3}, "host-client") in events, "sid resolved to the host's client"
    assert (1, b"\x89PNG\x00", None) in events, "bytes survived the JSON wire"
    assert ("send_progress_text", "working", "7", None) in host_server.sent
    # client_id was the host's, read inside the worker
    assert out == ["host-client"] or out == ("host-client",)
    previews = [d for (e, d, sid) in events if e == 2]
    if previews:  # PIL present in the test env
        fmt, img, max_size = previews[0]
        assert fmt == "JPEG" and max_size == 512 and img.size == (4, 4)


def test_send_sync_from_a_thread_is_dropped_not_interleaved(worker, host_server):
    out = _call(worker, "from_thread")
    assert out == ["ok"] or out == ("ok",)
    assert not any(e == "server_user.thread" for (k, e, d, sid) in host_server.sent if k == "send_sync")


def test_off_surface_use_fails_the_node_with_the_reason(worker):
    with pytest.raises(WorkerError) as ei:
        _call(worker, "off_surface")
    assert "prompt_queue is not available in an isolated node process" in str(ei.value)
