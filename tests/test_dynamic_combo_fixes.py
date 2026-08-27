"""Contract tests for the four dynamic-combo fixes (2026-08 review).

Each test pins one of the review's blocking findings:
B2  -- synthesized validate must be NAMED-ARG (a **kwargs form would disable
       execution.py's min/max clamps for every input on the node).
B3  -- synthesis is unconditional for marked inputs (3D-Pack never wrote one).
B5  -- fingerprint attaches only when safe (subset rule / explicit opt-in);
       a pack fingerprint over unmarked inputs must stay inert.
1c  -- author-supplied source dirs cannot escape the input root.
1d  -- an empty directory with no placeholder keeps the CACHED options.
"""

import inspect
import sys
import types

from comfy_env.isolation.metadata import (
    _contained_root,
    _make_named_validate,
    _make_dynamic_fingerprint,
    _scan_dynamic_dir,
    build_proxy_class,
)


def _fake_folder_paths(monkeypatch, input_dir):
    mod = types.ModuleType("folder_paths")
    mod.get_input_directory = lambda: str(input_dir)
    monkeypatch.setitem(sys.modules, "folder_paths", mod)


def _v1_meta(**over):
    meta = {
        "function": "run",
        "category": "test",
        "output_node": False,
        "return_types": ("STRING",),
        "return_names": (),
        "output_is_list": None,
        "input_is_list": None,
        "module_name": "fake.mod",
        "class_name": "FakeNode",
        "accelerator": None,
        "is_v3": False,
        "validate_args": None,
        "fingerprint_args": None,
        "input_types": {
            "required": {
                "mesh": (["a.obj"], {"comfy_env_dynamic_dir": "3d"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
            }
        },
    }
    meta.update(over)
    return meta


def _build(meta):
    return build_proxy_class(
        node_name="FakeNode", meta=meta, env_dir="/nonexistent",
        package_root="/nonexistent", sys_path=[],
        env_vars={}, health_check_timeout=1.0,
    )


# --- B2/B3: named-arg validate, synthesized unconditionally for marks -------

def test_validate_is_named_arg_and_covers_only_marked_inputs():
    cls = _build(_v1_meta())
    v = inspect.getattr_static(cls, "VALIDATE_INPUTS")
    assert isinstance(v, classmethod)
    spec = inspect.getfullargspec(v.__func__)
    assert spec.varkw is None, "**kwargs would disable min/max for EVERY input"
    assert spec.args == ["cls", "mesh"], "exempt exactly the marked combo"
    assert cls.VALIDATE_INPUTS(mesh="anything.obj") is True


def test_validate_union_includes_pack_declared_args():
    cls = _build(_v1_meta(validate_args=["mesh", "material"]))
    spec = inspect.getfullargspec(
        inspect.getattr_static(cls, "VALIDATE_INPUTS").__func__)
    assert spec.args == ["cls", "mesh", "material"]


def test_no_marks_no_pack_validate_means_no_synth():
    meta = _v1_meta()
    meta["input_types"] = {"required": {"steps": ("INT", {"default": 1})}}
    cls = _build(meta)
    assert inspect.getattr_static(cls, "VALIDATE_INPUTS", None) is None


def test_make_named_validate_rejects_non_identifiers():
    cm, names = _make_named_validate(["ok", "not-ok", "cls", "with space"])
    assert names == ["ok"]
    assert inspect.getfullargspec(cm.__func__).args == ["cls", "ok"]


# --- B5: fingerprint gate ---------------------------------------------------

def test_fingerprint_attaches_when_pack_args_subset_of_marks():
    cls = _build(_v1_meta(fingerprint_args=["mesh"]))
    assert inspect.getattr_static(cls, "IS_CHANGED", None) is not None


def test_fingerprint_stays_inert_for_wider_pack_args():
    # raytracer case: fingerprint over a LINKED input that is not a marked
    # combo. Attaching would poison its caching signature with None-stubs.
    cls = _build(_v1_meta(fingerprint_args=["cad_model", "resolution"]))
    assert inspect.getattr_static(cls, "IS_CHANGED", None) is None


def test_fingerprint_explicit_optin_marker():
    meta = _v1_meta()
    meta["input_types"]["required"]["mesh"] = (
        ["a.obj"], {"comfy_env_dynamic_dir": "3d",
                    "comfy_env_fingerprint": "mtime"})
    cls = _build(meta)
    assert inspect.getattr_static(cls, "IS_CHANGED", None) is not None


def test_fingerprint_changes_with_mtime_and_never_raises(tmp_path, monkeypatch):
    _fake_folder_paths(monkeypatch, tmp_path)
    (tmp_path / "3d").mkdir()
    f = tmp_path / "3d" / "m.obj"
    f.write_text("v1")
    cm = _make_dynamic_fingerprint(
        [("required", "mesh", {"comfy_env_dynamic_dir": "3d"})])
    class Host:
        fp = cm
    a = Host.fp(mesh="m.obj")
    import os as _os
    _os.utime(f, ns=(1, 1))
    b = Host.fp(mesh="m.obj")
    assert a != b, "mtime change must change the signature"
    # missing file: stable, not raising (an exception becomes NaN upstream)
    c1 = Host.fp(mesh="missing.obj")
    c2 = Host.fp(mesh="missing.obj")
    assert c1 == c2


# --- 1c: containment fence --------------------------------------------------

def test_absolute_dir_rejected(tmp_path, capsys):
    assert _contained_root(str(tmp_path), "/etc") is None
    assert "rejected" in capsys.readouterr().err


def test_traversal_rejected(tmp_path):
    assert _contained_root(str(tmp_path), "../..") is None


def test_plain_subdir_accepted(tmp_path):
    (tmp_path / "3d").mkdir()
    assert _contained_root(str(tmp_path), "3d") is not None


def test_scan_with_traversal_spec_keeps_cached(tmp_path, monkeypatch):
    _fake_folder_paths(monkeypatch, tmp_path)
    out = _scan_dynamic_dir({"comfy_env_sources": [{"dir": "../../evil"}]})
    assert out is None, "rejected source must fall back to cached options"


# --- 1d: empty dir sentinel -------------------------------------------------

def test_empty_dir_no_placeholder_returns_none(tmp_path, monkeypatch):
    _fake_folder_paths(monkeypatch, tmp_path)
    (tmp_path / "3d").mkdir()
    assert _scan_dynamic_dir({"comfy_env_dynamic_dir": "3d"}) is None


def test_empty_dir_with_placeholder_returns_it(tmp_path, monkeypatch):
    _fake_folder_paths(monkeypatch, tmp_path)
    (tmp_path / "3d").mkdir()
    out = _scan_dynamic_dir({"comfy_env_dynamic_dir": "3d",
                             "comfy_env_placeholder": "(none)"})
    assert out == ["(none)"]


def test_nonempty_dir_lists(tmp_path, monkeypatch):
    _fake_folder_paths(monkeypatch, tmp_path)
    (tmp_path / "3d").mkdir()
    (tmp_path / "3d" / "x.obj").write_text("")
    out = _scan_dynamic_dir({"comfy_env_dynamic_dir": "3d"})
    assert out == ["x.obj"]
