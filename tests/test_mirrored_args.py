"""Tests for the host CLI-arg mirror.

The pure functions run under bare CI with fake args objects; the ast guards
pin the two things CI cannot execute: the apply site's position relative to
the worker's first comfy import (late application is unrecoverable, because
the memory-relevant reads execute once at module import), and the attention
auto-probe staying behind its opt-in gate.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

from comfy_env.mirrored_args import (
    ATTENTION_KEY,
    MIRRORED_ARGS,
    apply_host_args,
    parse_denylist,
    readback_hash,
    resolve_host_attention,
    serialize_host_args,
    unmirrored_nondefault,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "comfy_env"
WORKER = SRC / "isolation" / "workers" / "_persistent_worker.py"
SUBPROCESS = SRC / "isolation" / "workers" / "subprocess.py"


def _fake_args(**kw):
    base = {name: False for name in MIRRORED_ARGS}
    base["async_offload"] = None
    base["fast"] = set()
    base.update(kw)
    return SimpleNamespace(**base)


class TestSerialize:
    def test_values_round_trip_through_json(self):
        args = _fake_args(fp8_e4m3fn_unet=True, disable_smart_memory=True,
                          async_offload=2)
        payload = json.loads(json.dumps(serialize_host_args(args)))
        target = _fake_args()
        out = apply_host_args(target, payload)
        assert target.fp8_e4m3fn_unet is True
        assert target.disable_smart_memory is True
        assert target.async_offload == 2
        assert "fp8_e4m3fn_unet" in out["applied"]

    def test_async_offload_none_stays_none(self):
        """comfy branches on `is not None` (model_management.py:1341): a None
        collapsed to 0 or "" by JSON handling would silently change the
        stream count. Catches: int() coercion on the wire."""
        payload = json.loads(json.dumps(serialize_host_args(_fake_args())))
        assert payload["async_offload"] is None
        target = _fake_args(async_offload=3)
        apply_host_args(target, payload)
        assert target.async_offload is None

    def test_non_allowlisted_attr_never_leaks_into_the_payload(self):
        """Catches: json.dumps(vars(args)) wholesale. The value check catches
        nested or renamed leakage the key check would miss."""
        args = _fake_args()
        args.canary_secret = "LEAK-hunter2"
        payload = serialize_host_args(args)
        assert "canary_secret" not in payload
        assert "LEAK-hunter2" not in json.dumps(payload)

    def test_denylist_withholds_named_flags_only(self):
        args = _fake_args(fast_disk=True, fp8_e4m3fn_unet=True)
        payload = serialize_host_args(args, deny=parse_denylist("fast_disk"))
        assert "fast_disk" not in payload
        assert payload["fp8_e4m3fn_unet"] is True

    def test_attention_is_the_resolved_backend_not_a_flag(self):
        """store_true makes host-False indistinguishable from host-default;
        the resolved name is what the worker must not upgrade past."""
        assert resolve_host_attention(
            SimpleNamespace(use_sage_attention=True,
                            use_flash_attention=False)) == "sage"
        assert resolve_host_attention(
            SimpleNamespace(use_sage_attention=False,
                            use_flash_attention=False)) is None
        payload = serialize_host_args(_fake_args())
        assert ATTENTION_KEY not in payload  # default host: no key at all


class TestApply:
    def test_hand_built_payload_cannot_smuggle_a_non_contract_flag(self):
        """The allowlist is enforced at APPLY too. Catches: an allowlist that
        only filters at serialize, letting a forged env var set anything."""
        target = _fake_args()
        target.canary_secret = "untouched"
        out = apply_host_args(target, {"canary_secret": "pwned",
                                       "fp16_unet": True})
        assert target.canary_secret == "untouched"
        assert any(s["name"] == "canary_secret"
                   and s["reason"] == "not_in_allowlist"
                   for s in out["skipped"])
        assert target.fp16_unet is True

    def test_unknown_flag_on_this_comfy_skips_never_raises(self):
        """Version skew: a payload naming a flag this ComfyUI lacks degrades,
        and never setattrs a name args does not have."""
        target = SimpleNamespace(fp16_unet=False)
        out = apply_host_args(target, {"fp16_unet": True,
                                       "disable_smart_memory": True})
        assert target.fp16_unet is True
        assert not hasattr(target, "disable_smart_memory")
        assert any(s["name"] == "disable_smart_memory"
                   and s["reason"] == "unknown_here" for s in out["skipped"])

    def test_fast_that_cannot_hydrate_skips_the_whole_flag(self):
        """Half a PerformanceFeature set would be an invented value; the
        whole flag skips instead. (Bare CI has no comfy, so hydration always
        fails here, which is exactly the failure path under test.)"""
        target = _fake_args()
        out = apply_host_args(target, {"fast": ["fp16_accumulation"]})
        assert target.fast == set()
        assert any(s["name"] == "fast"
                   and s["reason"].startswith("unhydratable")
                   for s in out["skipped"])

    def test_attention_key_is_not_an_args_dest(self):
        """The worker's attention site owns it (importability check); apply
        must neither setattr it nor report it as a defect."""
        target = _fake_args()
        out = apply_host_args(target, {ATTENTION_KEY: "sage"})
        assert not hasattr(target, ATTENTION_KEY)
        assert out["applied"] == [] and out["skipped"] == []


class TestReadbackHash:
    def test_hash_reads_the_args_object_not_the_payload(self):
        """The whole test: a payload-echo hash matches its source even when
        apply silently failed. Feed two args objects that diverge on one
        applied value and demand different hashes."""
        names = ["fp16_unet", "disable_smart_memory"]
        host = _fake_args(fp16_unet=True, disable_smart_memory=True)
        worker_ok = _fake_args(fp16_unet=True, disable_smart_memory=True)
        worker_drifted = _fake_args(fp16_unet=True,
                                    disable_smart_memory=False)
        assert readback_hash(host, names) == readback_hash(worker_ok, names)
        assert readback_hash(host, names) != readback_hash(worker_drifted,
                                                           names)

    def test_hash_is_order_invariant_and_set_stable(self):
        a = _fake_args(fast={"b", "a"})
        b = _fake_args(fast={"a", "b"})
        assert (readback_hash(a, ["fast", "fp16_unet"])
                == readback_hash(b, ["fp16_unet", "fast"]))

    def test_unmirrored_nondefault_names_the_vram_gap(self):
        args = _fake_args()
        args.novram = True
        args.lowvram = False
        assert unmirrored_nondefault(args) == ["novram"]


def _main_body_statements(tree):
    """Statements EXECUTED at main()'s body level: walk main() but skip any
    subtree rooted at a nested def or class. A lexical guard fails on
    today's code (comfy imports inside nested handlers sit above the apply
    site); executed-at-main-level is the property that matters."""
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    out = []
    stack = list(main.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            continue
        out.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return out


_REACHING_ROOTS = {"folder_paths", "nodes", "execution", "server"}


def _reaches_cli_args(module_name):
    if module_name is None:
        return False
    if module_name == "comfy.cli_args" or module_name == "comfy.options":
        return False  # cli_args is what apply mutates; options gates parsing
    root = module_name.split(".")[0]
    return root == "comfy" or root in _REACHING_ROOTS


def test_mirror_applies_before_any_executed_comfy_import():
    """The load-bearing ordering: DISABLE_SMART_MEMORY, NUM_STREAMS, the
    MAX_PINNED gate and DISABLE_MMAP are read ONCE at comfy module import,
    and `import folder_paths` transitively runs `from comfy.cli_args import
    args`, so an apply below the first reaching import is unrecoverable.
    Catches: the apply block drifting below the folder_paths import, or the
    guard rotting into a lexical check that nested defs fool."""
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    body = _main_body_statements(tree)

    import_hits = []
    for node in body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if _reaches_cli_args(a.name):
                    import_hits.append((node.lineno, a.name))
        elif isinstance(node, ast.ImportFrom):
            if _reaches_cli_args(node.module):
                import_hits.append((node.lineno, node.module))
    assert import_hits, (
        "guard found no reaching imports in main(); it stopped guarding")

    apply_calls = [n.lineno for n in body
                   if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "apply_host_args"]
    assert len(apply_calls) == 1, (
        f"expected exactly one executed apply_host_args call in main(), "
        f"found {len(apply_calls)}")
    first_import = min(l for l, _ in import_hits)
    assert apply_calls[0] < first_import, (
        f"apply_host_args at line {apply_calls[0]} runs AFTER the first "
        f"reaching import at line {first_import} "
        f"({dict(import_hits)[first_import]}); the mirrored flags freeze "
        f"before they are set")

    # And nothing at module top level may import a reaching module either.
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not any(_reaches_cli_args(a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not _reaches_cli_args(node.module)


def test_no_attention_autodetect_outside_the_gate():
    """The auto-probe guesses over the host's known answer, so it lives only
    inside the COMFY_ENV_WORKER_ATTENTION branch. Catches: a bare `import
    sageattention` probe reappearing anywhere else in the worker."""
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    gate_ifs = [n for n in ast.walk(tree)
                if isinstance(n, ast.If)
                and "_att_mode" in ast.unparse(n.test)]
    assert gate_ifs, "the attention gate is gone"
    spans = [(n.lineno, n.end_lineno) for n in gate_ifs]
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        for a in node.names:
            if a.name in ("sageattention", "flash_attn"):
                assert any(lo <= node.lineno <= hi for lo, hi in spans), (
                    f"attention probe import at line {node.lineno} sits "
                    f"outside the WORKER_ATTENTION gate")


def test_export_respects_pack_env_vars_and_the_kill_switch():
    """The env write must be guarded not-in-env (pack [env_vars] wins) and
    behind the global kill, and must serialize via serialize_host_args,
    never argv or vars(args)."""
    src = SUBPROCESS.read_text(encoding="utf-8")
    assert "MIRROR_ENV_VAR not in env" in src
    assert "MIRROR_KILL_ENV_VAR" in src
    assert "serialize_host_args(_ma_args" in src
    fn_start = src.index("def _ensure_started")
    fn_src = src[fn_start:src.index("\n    def ", fn_start + 10)]
    assert "vars(_ma_args)" not in fn_src and "sys.argv" not in fn_src
