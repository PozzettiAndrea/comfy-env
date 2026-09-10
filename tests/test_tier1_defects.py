"""Regression pins for defects that were silent in production.

Each of these failed the same way: the code did something wrong and reported
success, so nothing upstream could notice.
"""

import json
import os
import sys
import types

import pytest


# --------------------------------------------------------------------------
# validate_env_stamp: a corrupt stamp must not take the pack out of ComfyUI
# --------------------------------------------------------------------------

@pytest.mark.parametrize("content", ["null", '"a string"', "[1, 2]"])
def test_non_object_stamp_does_not_raise(tmp_path, content):
    """json.loads succeeds for these; .get() does not.

    The AttributeError escaped _resolve_env_dir into register_nodes(), so one
    corrupt byte removed every node in the pack with an unrelated traceback.
    """
    from comfy_env.environment.cache import validate_env_stamp

    (tmp_path / "env.stamp.json").write_text(content, encoding="utf-8")
    ok, reason = validate_env_stamp(tmp_path)   # must not raise
    assert ok is True
    assert "not verified" in reason


def test_matching_stamp_still_verifies(tmp_path):
    from comfy_env.environment.cache import _abi_tag, validate_env_stamp

    (tmp_path / "env.stamp.json").write_text(
        json.dumps({"abi_tag": _abi_tag()}), encoding="utf-8")
    ok, _ = validate_env_stamp(tmp_path)
    assert ok is True


def test_foreign_stamp_is_still_refused(tmp_path):
    """The check must keep doing its actual job."""
    from comfy_env.environment.cache import validate_env_stamp

    (tmp_path / "env.stamp.json").write_text(
        json.dumps({"abi_tag": "py39-torch1-13-cu117"}), encoding="utf-8")
    ok, reason = validate_env_stamp(tmp_path)
    assert ok is False
    assert "py39-torch1-13-cu117" in reason


# --------------------------------------------------------------------------
# COMFY_ENV_ROOT
# --------------------------------------------------------------------------

def test_comfy_env_root_expands_tilde(monkeypatch, tmp_path):
    """`COMFY_ENV_ROOT=~/envs` from a Dockerfile/systemd/CI has no shell to
    expand it; Path('~/envs') creates a literal '~' directory under cwd."""
    import comfy_env.environment.cache as cache

    # Point HOME at tmp_path FIRST. `_short_global_root` mkdirs what it
    # resolves, so without this the test creates `~/ce-test-root` in the
    # developer's real home directory and never removes it.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("COMFY_ENV_ROOT", "~/ce-test-root")
    monkeypatch.setattr(cache, "_ANNOUNCED_WS", True, raising=False)
    root = cache._short_global_root()

    assert "~" not in str(root), f"tilde left unexpanded: {root}"
    assert root.is_absolute()
    assert str(root).startswith(str(tmp_path))


def test_comfy_env_root_is_absolute(monkeypatch, tmp_path):
    """A relative value would resolve differently per working directory."""
    import comfy_env.environment.cache as cache

    # chdir into tmp_path first: `_short_global_root` resolves a relative
    # value against the CWD and then MKDIRS it, so run from the repo root
    # this test left an empty `relative-envs/` directory in the working tree.
    # Git never reported it, because git does not track empty directories.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMFY_ENV_ROOT", "relative-envs")
    monkeypatch.setattr(cache, "_ANNOUNCED_WS", True, raising=False)
    root = cache._short_global_root()
    assert root.is_absolute()
    assert str(root).startswith(str(tmp_path))


# --------------------------------------------------------------------------
# libomp: the guard matched every path, so nothing was ever deduped
# --------------------------------------------------------------------------

def _fake_env(tmp_path, abi):
    sp = tmp_path / f"geometrypack-{abi}" / "lib" / "python3.13" / "site-packages"
    (sp / "torch" / "lib").mkdir(parents=True)
    (sp / "torch" / "lib" / "libomp.dylib").write_text("canonical")
    for pkg, sub in (("pymeshlab", "Frameworks"), ("sklearn", ".dylibs")):
        (sp / pkg / sub).mkdir(parents=True)
        (sp / pkg / sub / "libomp.dylib").write_text("duplicate")
    # conda-forge's copy at <env>/lib/libomp.dylib -- a real candidate the
    # globs are written to catch.
    (sp.parent.parent / "libomp.dylib").write_text("duplicate")
    return sp


@pytest.mark.parametrize("abi", ["py313-torch2-13-cpu", "py313-notorch"])
def test_dedupe_links_duplicates_regardless_of_env_name(tmp_path, monkeypatch, abi):
    """`if "torch" in libomp` was a substring test on the FULL PATH.

    Every env directory is named <pack>-<abi_tag>, and _abi_tag() emits either
    "torchN-M" or "notorch" -- both contain "torch". So every candidate was
    classified as torch's own and dedupe_libomp had never linked anything on
    any machine.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    from comfy_env.environment import libomp as L

    sp = _fake_env(tmp_path, abi)
    result = L.dedupe_libomp(site_packages=sp)

    assert result.linked == 3, f"nothing deduped: {result.summary()}"
    assert result.skipped == 1, "the canonical torch libomp must be the only skip"
    for pkg, sub in (("pymeshlab", "Frameworks"), ("sklearn", ".dylibs")):
        assert os.path.islink(sp / pkg / sub / "libomp.dylib")


def test_dedupe_restores_the_original_when_symlink_fails(tmp_path, monkeypatch):
    """rename-then-symlink with no rollback left a package with NO libomp."""
    monkeypatch.setattr(sys, "platform", "darwin")
    from comfy_env.environment import libomp as L

    sp = _fake_env(tmp_path, "py313-torch2-13-cpu")
    victim = sp / "pymeshlab" / "Frameworks" / "libomp.dylib"

    def boom(src, dst):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(L.os, "symlink", boom)
    result = L.dedupe_libomp(site_packages=sp)

    assert victim.exists(), "package was left with no libomp.dylib at all"
    assert victim.read_text() == "duplicate"
    assert result.failed > 0
    assert result.status != "ok", "a failed dedupe must not report status ok"


# --------------------------------------------------------------------------
# cancel
# --------------------------------------------------------------------------

def test_cancel_raises_so_the_worker_hears_about_it(monkeypatch):
    """InterruptProcessingException is a BaseException.

    `except Exception` never caught it, so it unwound out of _send_request
    mid-conversation while the worker sat blocked in _call_parent's recv().
    And returning an error DICT was invisible: _handle_callback wraps any
    return as {"status": "ok", "result": ...} and the worker checks only the
    outer status. It has to raise.
    """
    from comfy_env.isolation import pool

    class InterruptProcessingException(BaseException):
        pass

    mm = types.ModuleType("comfy.model_management")
    mm.InterruptProcessingException = InterruptProcessingException

    # The flag, and both of upstream's accessors. The throwing one CLEARS it
    # before raising; the reader does not. Which one comfy-env calls decides
    # whether the user's click survives to be seen again.
    state = {"flag": True, "consumed": False}

    def processing_interrupted():
        return state["flag"]

    def throw_exception_if_processing_interrupted():
        if state["flag"]:
            state["flag"] = False
            state["consumed"] = True
            raise InterruptProcessingException()

    mm.processing_interrupted = processing_interrupted
    mm.throw_exception_if_processing_interrupted = \
        throw_exception_if_processing_interrupted
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    with pytest.raises(RuntimeError, match="interrupted"):
        pool._handle_progress({"value": 1, "total": 2})

    # The click must still be there. Consuming it here meant that a pack
    # swallowing the exception spent the user's Stop for nothing, and they
    # had to press it again.
    assert state["flag"] is True, "the interrupt flag was consumed"
    assert state["consumed"] is False, \
        "comfy-env called the CONSUMING accessor"


def test_no_comfyui_is_not_reported_as_a_user_cancel(monkeypatch):
    """Outside ComfyUI the ImportError used to be reported as 'interrupted'."""
    from comfy_env.isolation import pool

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def no_comfy(name, *a, **kw):
        if name.startswith("comfy"):
            raise ImportError("no comfy here")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", no_comfy)
    assert pool._handle_progress({"value": 1, "total": 2}) == {}


# --------------------------------------------------------------------------
# metadata: the loud warning must survive a cache hit
# --------------------------------------------------------------------------

def test_empty_v3_scan_warns_on_every_startup_not_just_the_first():
    """A zero-node payload is cached like any other and only invalidates on a
    .py mtime change, so warning solely on the fresh-scan path meant a broken
    pack screamed once and was silent forever after."""
    import inspect

    from comfy_env.isolation import metadata

    src = inspect.getsource(metadata.fetch_metadata)
    before_return = src.split("return payload")[0]
    assert "_warn_empty_v3_scan" in before_return, (
        "the cache-hit path returns before warning, so a broken pack is "
        "reported once and never again"
    )


# --------------------------------------------------------------------------
# validate exemption: **kwargs
# --------------------------------------------------------------------------

def test_a_kwargs_validate_reproduces_the_exemption():
    """ComfyUI exempts an input if it is NAMED or the function has a catch-all.

    Capturing only the names turned `validate_inputs(cls, **kwargs)` -- which
    asks to exempt everything -- into no validate at all, so ComfyUI re-imposed
    every check the author waived and a workflow that submits fine natively was
    REJECTED once the pack was isolated.
    """
    import inspect

    from comfy_env.isolation.metadata import _make_named_validate

    cm, _ = _make_named_validate([], varkw=True)
    assert cm is not None, "a **kwargs validate must still attach"
    spec = inspect.getfullargspec(cm.__func__)
    assert spec.varkw is not None, "the catch-all must survive"

    # the mixed form keeps both halves
    cm, _ = _make_named_validate(["model_file"], varkw=True)
    spec = inspect.getfullargspec(cm.__func__)
    assert "model_file" in spec.args and spec.varkw is not None

    # and a named-only original must NOT gain one: that would exempt every
    # input including the numeric clamps users rely on
    cm, _ = _make_named_validate(["model_file"], varkw=False)
    assert inspect.getfullargspec(cm.__func__).varkw is None


# --------------------------------------------------------------------------
# node state: the gate and the wire must be one predicate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value", [
    {1: "a"},            # int keys -> came back as strings
    (1, 2, 3),           # tuple -> came back as a list
    {"s"},               # set: passed the pickle gate, died on the wire
    b"bytes",            # same
    [1, {"k": (2, 3)}],  # nested tuple
])
def test_state_values_round_trip_with_their_types(value):
    """The gate asked pickle and the wire was JSON, so a picklable-but-not-JSON
    value was promised as shippable and then killed the call. Values that DID
    cross were silently retyped."""
    from comfy_env import state_sync

    wire = state_sync.encode_value(value)
    json.dumps(wire)  # must survive the actual transport
    assert state_sync.decode_value(wire) == value
    assert type(state_sync.decode_value(wire)) is type(value)


def test_json_native_state_is_not_wrapped():
    """The common frame must stay readable, not become opaque envelopes."""
    from comfy_env import state_sync

    for v in ("s", 1, 1.5, True, None, {"a": [1, 2]}):
        assert state_sync.encode_value(v) == v


# --------------------------------------------------------------------------
# a coroutine is not a serialization problem
# --------------------------------------------------------------------------

def test_a_coroutine_does_not_blame_the_serializer():
    """The generic pickle message sent authors to write a serializer for a
    type no serializer can help with. The real cause is an un-awaited
    `async def`."""
    from comfy_env.isolation.workers import _ipc_shared

    async def _node():
        return 1

    coro = _node()
    try:
        with pytest.raises(TypeError) as excinfo:
            _ipc_shared._to_shm_generic(
                coro, {}, {}, tensor_serializer=None)
        msg = str(excinfo.value)
        assert "async def" in msg
        assert "serialization.py" not in msg, "still blaming the serializer"
        assert "do not add a serializer" in msg.lower()
    finally:
        coro.close()


# --------------------------------------------------------------------------
# forwarded previews
# --------------------------------------------------------------------------

def test_a_forwarded_preview_rebuilds_into_upstreams_tuple():
    """The worker cannot put a PIL image on a JSON frame, so it ships
    (format, base64, max_size); the parent owes upstream a real
    PreviewImageTuple."""
    PIL = pytest.importorskip("PIL.Image")
    import base64
    import io

    from comfy_env.isolation import pool

    buf = io.BytesIO()
    PIL.new("RGB", (4, 4), (255, 0, 0)).save(buf, format="PNG")
    wire = ["PNG", base64.b64encode(buf.getvalue()).decode("ascii"), 512]

    fmt, img, max_size = pool._decode_preview(wire)
    assert (fmt, max_size) == ("PNG", 512)
    assert img.size == (4, 4)


def test_a_broken_preview_never_costs_the_progress_tick():
    """A preview is availability, not correctness."""
    from comfy_env.isolation import pool

    assert pool._decode_preview(None) is None
    assert pool._decode_preview(["PNG", "not-base64!!", 512]) is None
