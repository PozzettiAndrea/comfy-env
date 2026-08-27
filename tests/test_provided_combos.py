"""Contract tests for the provider mechanism (phase-3 composite).

The stenographer's output side: ProvidedList tag math, input_files, the
parent resolver (fencing, membership gate, private registry, mtime cache),
mark precedence (journal wins over legacy markers), and splice affixes.
"""

import os
import sys
import types

from comfy_env.isolation.provided import ProvidedList, input_files
import comfy_env.isolation.metadata as md


def _fake_fp(monkeypatch, input_dir, registry=None):
    mod = types.ModuleType("folder_paths")
    mod.get_input_directory = lambda: str(input_dir)
    mod.folder_names_and_paths = registry if registry is not None else {}
    def gfl(name):
        paths, exts = mod.folder_names_and_paths[name]
        out = []
        for p in paths:
            for f in sorted(os.listdir(p)):
                if not exts or os.path.splitext(f)[1] in exts:
                    out.append(f)
        return out
    mod.get_filename_list = gfl
    def rs(p, excluded_dir_names=None):
        files = []
        for r, _d, ff in os.walk(p):
            for f in ff:
                files.append(os.path.relpath(os.path.join(r, f), p))
        return files, {}
    mod.recursive_search = rs
    mod.filter_files_extensions = lambda files, exts: (
        files if not exts else [f for f in files if os.path.splitext(f)[1] in exts])
    monkeypatch.setitem(sys.modules, "folder_paths", mod)
    return mod


# --- tag math ---------------------------------------------------------------

def test_radd_carries_provider_and_offset():
    g = ProvidedList(["a", "b"], provider={"kind": "x"})
    r = ["none"] + g
    assert isinstance(r, ProvidedList)
    assert (r.offset, r.span) == (1, 2) and r == ["none", "a", "b"]
    r2 = r + ["tail"]
    assert (r2.offset, r2.span) == (1, 2)


def test_sorted_sheds_the_tag():
    g = ProvidedList(["b", "a"], provider={"kind": "x"})
    assert not isinstance(sorted(g), ProvidedList)


# --- input_files ------------------------------------------------------------

def test_input_files_lists_and_declares(tmp_path, monkeypatch):
    _fake_fp(monkeypatch, tmp_path)
    (tmp_path / "3d").mkdir()
    (tmp_path / "3d" / "m.obj").write_text("")
    (tmp_path / "flat.obj").write_text("")
    (tmp_path / "skip.txt").write_text("")
    out = input_files(
        [{"dir": "3d", "recursive": True, "rel_to_input": True}, ""],
        exts=[".obj"], placeholder="(none)")
    assert out == ["3d/m.obj", "flat.obj"]
    assert out.provider["kind"] == "input_dir"
    assert out.provider["placeholder"] == "(none)"


def test_input_files_placeholder_when_empty(tmp_path, monkeypatch):
    _fake_fp(monkeypatch, tmp_path)
    out = input_files(["nothing_here"], exts=[".obj"], placeholder="(none)")
    assert out == ["(none)"] and out.provider["sources"][0]["dir"] == "nothing_here"


# --- resolver ---------------------------------------------------------------

def test_resolver_input_dir_lists_and_caches(tmp_path, monkeypatch):
    _fake_fp(monkeypatch, tmp_path)
    (tmp_path / "3d").mkdir()
    (tmp_path / "3d" / "a.obj").write_text("")
    prov = {"kind": "input_dir",
            "sources": [{"dir": "3d", "recursive": True, "rel_to_input": True}],
            "exts": [".obj"], "placeholder": None}
    md._LIVE_CACHE.clear()
    assert md._resolve_provider(prov) == ["3d/a.obj"]
    # cache hit path (mtimes unchanged) returns the same
    assert md._resolve_provider(prov) == ["3d/a.obj"]
    # change invalidates. Force a distinct timestamp: the whole test runs
    # inside one filesystem timestamp granule, which is exactly the race the
    # ns compare narrows -- the test must not depend on winning it.
    (tmp_path / "3d" / "b.obj").write_text("")
    os.utime(tmp_path / "3d", ns=(1, 1))
    assert md._resolve_provider(prov) == ["3d/a.obj", "3d/b.obj"]


def test_resolver_membership_gate_no_keyerror(monkeypatch, tmp_path):
    _fake_fp(monkeypatch, tmp_path, registry={})
    assert md._resolve_provider(
        {"kind": "filename_list", "category": "nope"}) is None


def test_resolver_core_category_read_only(monkeypatch, tmp_path):
    mdir = tmp_path / "checkpoints"; mdir.mkdir()
    (mdir / "x.ckpt").write_text("")
    fp = _fake_fp(monkeypatch, tmp_path,
                  registry={"checkpoints": ([str(mdir)], {".ckpt"})})
    before = dict(fp.folder_names_and_paths)
    assert md._resolve_provider(
        {"kind": "filename_list", "category": "checkpoints"}) == ["x.ckpt"]
    assert fp.folder_names_and_paths == before, "registry must never be mutated"


def test_resolver_private_registry_host_wins(monkeypatch, tmp_path):
    packdir = tmp_path / "packmodels"; packdir.mkdir()
    (packdir / "p.bin").write_text("")
    hostdir = tmp_path / "host"; hostdir.mkdir()
    (hostdir / "h.ckpt").write_text("")
    fp = _fake_fp(monkeypatch, tmp_path,
                  registry={"shared": ([str(hostdir)], {".ckpt"})})
    md._PACK_FOLDER_REGISTRY.clear()
    md._register_pack_folders({"shared": {"paths": [str(packdir)], "exts": [".bin"]},
                               "private": {"paths": [str(packdir)], "exts": [".bin"]}})
    # host-defined name -> host's listing, pack's paths ignored here
    assert md._resolve_provider(
        {"kind": "filename_list", "category": "shared"}) == ["h.ckpt"]
    # pack-only name -> private replay via pure helpers
    assert md._resolve_provider(
        {"kind": "filename_list", "category": "private"}) == ["p.bin"]
    assert "private" not in fp.folder_names_and_paths


def test_resolver_never_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", None)
    assert md._resolve_provider({"kind": "input_dir", "sources": []}) is None
    assert md._resolve_provider({"kind": "filename_list", "category": "x"}) is None
    assert md._resolve_provider(None) is None


# --- marks precedence + splice affixes -------------------------------------

def test_journal_wins_over_legacy_marker():
    input_types = {"required": {
        "mesh": (["old.obj"], {"comfy_env_dynamic_dir": "3d"}),
    }}
    volatile = [{"section": "required", "name": "mesh",
                 "provider": {"kind": "input_dir", "sources": []},
                 "prefix": [], "suffix": []}]
    marks = md._collect_dynamic_marks(input_types, volatile)
    assert len(marks) == 1 and "__provider__" in marks[0][2], \
        "one resolver per input: the journal silences the legacy marker"


def test_splice_preserves_prefix_and_suffix(monkeypatch, tmp_path):
    _fake_fp(monkeypatch, tmp_path)
    (tmp_path / "3d").mkdir()
    (tmp_path / "3d" / "new.obj").write_text("")
    md._LIVE_CACHE.clear()
    sections = {"required": {"mesh": (["none", "stale.obj", "END"], {})}}
    marks = [("required", "mesh", {
        "__provider__": {"kind": "input_dir",
                         "sources": [{"dir": "3d", "recursive": False,
                                      "rel_to_input": False}],
                         "exts": [".obj"], "placeholder": None},
        "__prefix__": ["none"], "__suffix__": ["END"]})]
    out = md._splice_dynamic_options(sections, marks)
    assert out["required"]["mesh"][0] == ["none", "new.obj", "END"]
