"""Contract: an isolated node's dropdowns are live when a worker can answer.

comfy-env used to keep dropdowns fresh by capturing the RECIPE that built
them -- a journal of `folder_paths` calls, a tagged list subclass, offset
arithmetic to preserve a `["none"] +` prefix, and a helper packs had to
import. It bound about a sixth of the ecosystem and could not see a
hand-rolled `os.walk` at all.

It is replaced by a three-rung ladder (`_refresh_combo_options`): ask the
env's worker to re-run the node's own INPUT_TYPES if that worker is alive
AND idle; otherwise use the options captured at scan time. No guessing --
it is the pack's real function, in the env that owns it.

The property that makes the ladder safe is that its miss path is exactly
the old frozen behaviour, so it can only ever add. These tests pin that as
hard as they pin the hit path.
"""

import inspect

import pytest

from comfy_env.isolation.metadata import (
    _combo_input_names,
    _make_named_validate,
    _refresh_combo_options,
    _splice_combo_options,
    build_proxy_class,
)


# --- the ladder -------------------------------------------------------------

class _Worker:
    """Stands in for SubprocessWorker's no-spawn command surface."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def send_command_no_spawn(self, method, lock_timeout=None, **params):
        self.calls.append((method, lock_timeout, params))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture()
def pool(monkeypatch):
    """Swap the worker pool for a dict this test owns."""
    import comfy_env.isolation.pool as pool_mod
    fake = {}
    monkeypatch.setattr(pool_mod, "_WORKER_POOL", fake, raising=False)
    return fake


FRESH = {"options": {"required": {"mesh": ["a.obj", "b.obj"]}}, "status": "ok"}


def test_rung_1_alive_and_idle_answers(pool):
    pool["/env"] = (_Worker(FRESH), 1)
    assert _refresh_combo_options("/env", "m", "C") == FRESH["options"]


def test_rung_2_busy_falls_through(pool):
    # send_command_no_spawn returns the sentinel string rather than blocking.
    pool["/env"] = (_Worker("busy"), 1)
    assert _refresh_combo_options("/env", "m", "C") is None


def test_rung_2_dead_falls_through(pool):
    pool["/env"] = (_Worker("dead"), 1)
    assert _refresh_combo_options("/env", "m", "C") is None


def test_rung_3_no_worker_never_spawns_one(pool):
    # The pool is empty and must stay empty: /object_info enumerates every
    # registered node, so spawning here would start every env on the machine.
    assert _refresh_combo_options("/env", "m", "C") is None
    assert pool == {}


def test_a_raising_worker_is_not_a_raising_dropdown(pool):
    # A raise inside INPUT_TYPES makes core omit the node from /object_info
    # entirely. A stale dropdown is strictly better than a vanished node.
    pool["/env"] = (_Worker(RuntimeError("socket died")), 1)
    assert _refresh_combo_options("/env", "m", "C") is None


def test_worker_error_reply_falls_through(pool):
    pool["/env"] = (_Worker({"status": "error", "error": "boom"}), 1)
    assert _refresh_combo_options("/env", "m", "C") is None


def test_the_lock_wait_is_short(pool):
    # Cosmetic work on the /object_info path must never stall ComfyUI: if we
    # lose the race the answer is "use the cache", which is where we started.
    w = _Worker(FRESH)
    pool["/env"] = (w, 1)
    _refresh_combo_options("/env", "m", "C")
    (method, lock_timeout, params) = w.calls[0]
    assert method == "refresh_input_types"
    assert 0 < lock_timeout <= 0.5
    assert params == {"module": "m", "class_name": "C"}


# --- the splice -------------------------------------------------------------

CACHED = {
    "required": {
        "mesh": (["old.obj"], {"tooltip": "pick one", "image_upload": True}),
        "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
    }
}


def test_options_are_replaced():
    out = _splice_combo_options(CACHED, {"required": {"mesh": ["new.obj"]}})
    assert out["required"]["mesh"][0] == ["new.obj"]


def test_the_config_dict_survives():
    # Everything except the options list is still the scan-time capture:
    # tooltips, defaults and image_upload are not the worker's to change.
    out = _splice_combo_options(CACHED, {"required": {"mesh": ["new.obj"]}})
    assert out["required"]["mesh"][1] == {"tooltip": "pick one", "image_upload": True}


def test_an_empty_listing_keeps_the_cache():
    # Splicing [] would fail combo validation for every saved workflow using
    # the node the moment a folder is empty -- mid-upload, drive unmounted.
    out = _splice_combo_options(CACHED, {"required": {"mesh": []}})
    assert out["required"]["mesh"][0] == ["old.obj"]


def test_a_non_combo_is_left_alone():
    out = _splice_combo_options(CACHED, {"required": {"steps": ["nonsense"]}})
    assert out["required"]["steps"] == ("INT", {"default": 20, "min": 1, "max": 100})


def test_an_unreported_input_is_left_alone():
    # The fresh answer is additive. Silence about an input is not a claim
    # that it disappeared.
    out = _splice_combo_options(CACHED, {"required": {"mesh": ["new.obj"]}})
    assert "steps" in out["required"]


# --- which inputs the exemption covers --------------------------------------

def test_combo_inputs_are_found_and_others_are_not():
    assert _combo_input_names(CACHED) == ["mesh"]


def _v1_meta(**over):
    meta = {
        "function": "run", "category": "test", "output_node": False,
        "return_types": ("STRING",), "return_names": (), "output_is_list": None,
        "input_is_list": None, "module_name": "fake.mod", "class_name": "FakeNode",
        "accelerator": None, "is_v3": False, "validate_args": None,
        "fingerprint_args": None, "input_types": CACHED,
    }
    meta.update(over)
    return meta


def _build(meta):
    return build_proxy_class(
        node_name="FakeNode", meta=meta, env_dir="/nonexistent",
        package_root="/nonexistent", sys_path=[],
        env_vars={}, health_check_timeout=1.0,
    )


def test_validate_exempts_combos_and_only_combos():
    # Options can now change under core between the dropdown being drawn and
    # the prompt being validated, so the membership check has to be relaxed
    # for combos. It must NOT be relaxed for `steps`, whose min/max is real.
    cls = _build(_v1_meta())
    v = inspect.getattr_static(cls, "VALIDATE_INPUTS")
    assert isinstance(v, classmethod)
    spec = inspect.getfullargspec(v.__func__)
    assert spec.varkw is None, "**kwargs would disable min/max for EVERY input"
    assert spec.args == ["cls", "mesh"]
    assert cls.VALIDATE_INPUTS(mesh="anything.obj") is True


def test_validate_union_includes_pack_declared_args():
    cls = _build(_v1_meta(validate_args=["mesh", "material"]))
    spec = inspect.getfullargspec(
        inspect.getattr_static(cls, "VALIDATE_INPUTS").__func__)
    assert spec.args == ["cls", "mesh", "material"]


def test_no_combos_and_no_pack_validate_means_no_synth():
    meta = _v1_meta()
    meta["input_types"] = {"required": {"steps": ("INT", {"default": 1})}}
    cls = _build(meta)
    assert inspect.getattr_static(cls, "VALIDATE_INPUTS", None) is None


def test_make_named_validate_rejects_non_identifiers():
    cm, names = _make_named_validate(["ok", "not-ok", "cls", "with space"])
    assert names == ["ok"]
    assert inspect.getfullargspec(cm.__func__).args == ["cls", "ok"]


# --- the proxy end to end ---------------------------------------------------

def test_input_types_returns_the_cache_when_nothing_answers(pool):
    cls = _build(_v1_meta())
    assert cls.INPUT_TYPES()["required"]["mesh"][0] == ["old.obj"]


def test_input_types_goes_live_when_a_worker_answers(pool):
    pool["/nonexistent"] = (_Worker(FRESH), 1)
    cls = _build(_v1_meta())
    got = cls.INPUT_TYPES()
    assert got["required"]["mesh"][0] == ["a.obj", "b.obj"]
    assert got["required"]["steps"] == ("INT", {"default": 20, "min": 1, "max": 100})


def test_input_files_is_gone_from_the_public_api():
    # It was the one comfy-env-ism a pack had to import to get a live
    # input/ dropdown. The ladder runs the pack's real function instead,
    # so there is nothing left for an author to call (ADR-0017: the old
    # way goes in the same release).
    import comfy_env
    assert "input_files" not in comfy_env.__all__
    with pytest.raises(AttributeError):
        comfy_env.input_files
