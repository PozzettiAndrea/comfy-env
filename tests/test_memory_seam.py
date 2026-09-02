"""Contracts for the memory seam between ComfyUI and an isolation worker.

Every assertion here encodes something that was got WRONG during the audit that
produced this file, so each one is a regression guard rather than a restatement
of the code.

These are deliberately source level, using ``ast`` and text matching rather than
imports. The existing surface canary skips whenever ComfyUI is not importable,
which is precisely the environment CI runs in, so an import based guard on this
seam would never execute. A guard that does not run is worse than no guard,
because it reads as coverage.
"""

import ast
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src" / "comfy_env"
POOL = SRC / "isolation" / "pool.py"
WORKER = SRC / "isolation" / "workers" / "_persistent_worker.py"
PROXY = SRC / "isolation" / "model_patcher.py"
MEMMGR = SRC / "memory_manager.py"


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


class TestProxyRegistration:
    """The proxy in ComfyUI's ledger is the whole reason a worker's models are
    visible. Losing it would make them invisible AND unevictable, silently."""

    def test_proxy_is_inserted_into_current_loaded_models(self):
        src = POOL.read_text(encoding="utf-8")
        assert "current_loaded_models.insert(" in src, (
            "comfy-env no longer inserts a LoadedModel into ComfyUI's ledger. "
            "Without it a worker's models are invisible to the host's memory "
            "manager and nothing can evict them."
        )

    def test_registration_still_bypasses_load_models_gpu(self):
        """It inserts a LoadedModel directly and must keep doing so: calling
        load_models_gpu would try to load every model at once and OOM."""
        src = POOL.read_text(encoding="utf-8")
        insert_line = next(
            line for line in src.splitlines()
            if "current_loaded_models.insert(" in line
        )
        assert "LoadedModel" in src, "LoadedModel wrapper is gone"
        assert "load_models_gpu" not in insert_line


class TestEvictionSemantics:
    """The eviction path's arguments decide whether the HOST's own models can be
    evicted. Getting this backwards was one of the audit's findings."""

    def test_free_memory_is_not_called_with_for_dynamic(self):
        """`for_dynamic=True` activates the bypass at model_management.py:884,
        which makes every dynamic host model unevictable. comfy-env must leave
        it at its default so the host's models stay candidates. Checked on the
        ast so a multi line call cannot slip past a line grep."""
        offenders = [
            f"{POOL.name}:{node.lineno}"
            for node in ast.walk(_tree(POOL))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "free_memory"
            and any(k.arg == "for_dynamic" for k in node.keywords)
        ]
        assert not offenders, (
            f"comfy-env passes for_dynamic to free_memory at {offenders}, "
            f"enabling the eviction bypass for host dynamic models."
        )


class TestNodeBoundaryRelease:
    """ComfyUI releases per-node buffers in a `finally` around every node
    (execution.py:550). A worker never runs that code, so comfy-env mirrors it.
    If it is not in a `finally`, a raising node leaks."""

    def _release_calls_inside_finally(self) -> int:
        tree = _tree(WORKER)
        found = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try) or not node.finalbody:
                continue
            for stmt in node.finalbody:
                for sub in ast.walk(stmt):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "release_node_boundary"
                    ):
                        found += 1
        return found

    def test_release_runs_in_a_finally(self):
        assert self._release_calls_inside_finally() >= 1, (
            "release_node_boundary is not called from a finally block. A node "
            "that raises would then leak its per-node buffers."
        )

    def test_both_worker_call_paths_are_covered(self):
        """call_method and call_function are both one node execution."""
        assert self._release_calls_inside_finally() >= 2, (
            "only one worker call path releases at the node boundary; both "
            "call_method and call_function need it."
        )


class TestNoRoutesOnComfyUI:
    """Registering an aiohttp route from register_nodes crashed ComfyUI on the
    second isolated pack: ComfyUI flushes ONE shared RouteTableDef, so the
    duplicate raised `Added route will never be executed`. It also violates the
    goal that comfy-env stay invisible to upstream.

    `_register_proxy_routes` is exempt: it is per pack and fires only for packs
    that declare ROUTES, so it is user intended, and exempted by FUNCTION SCOPE
    rather than by line so a refactor inside it cannot trip this."""

    def test_comfy_env_registers_no_global_route(self):
        offenders = []
        for path in SRC.rglob("*.py"):
            src = path.read_text(encoding="utf-8", errors="replace")
            if "PromptServer" not in src:
                continue
            tree = ast.parse(src)
            exempt_ranges = [
                (n.lineno, max(x.lineno for x in ast.walk(n) if hasattr(x, "lineno")))
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_register_proxy_routes"
            ]

            def exempt(lineno):
                return any(a <= lineno <= b for a, b in exempt_ranges)

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or exempt(node.lineno):
                    continue
                # decorator/direct spelling: routes.get(...) / routes.post(...)
                f = node.func
                if (isinstance(f, ast.Attribute) and f.attr in ("get", "post")
                        and isinstance(f.value, ast.Name) and f.value.id == "routes"):
                    offenders.append(f"{path.name}:{node.lineno}")
                # getattr spelling: getattr(<...>.routes, method...)
                if (isinstance(f, ast.Name) and f.id == "getattr" and node.args
                        and isinstance(node.args[0], ast.Attribute)
                        and node.args[0].attr == "routes"):
                    offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, (
            f"comfy-env registers routes on ComfyUI's shared route table at "
            f"{offenders}. register_nodes runs once per pack, so a second "
            f"isolated pack raises RuntimeError before ComfyUI starts."
        )


class TestWorkerManagerReporting:
    """A worker never runs main.py, so it resolves to the legacy ledger while
    the host is normally on aimdo. That was invisible before this shipped."""

    def test_ready_frame_carries_the_manager(self):
        src = WORKER.read_text(encoding="utf-8")
        ready = [ln for ln in src.splitlines() if '"status": "ready"' in ln]
        assert ready, "ready frame not found"
        assert any("memory_manager" in ln for ln in ready), (
            "the ready frame no longer reports which memory manager the worker "
            "resolved to; the host cannot detect a mismatch without it."
        )

    def test_parent_captures_it(self):
        src = (SRC / "isolation" / "workers" / "subprocess.py").read_text(encoding="utf-8")
        assert "self.memory_manager" in src


class TestAimdoIsDefeatable:
    """Worker aimdo is on by default so a worker matches its host. It is a real
    behaviour change, so it must stay switchable off without editing code, and
    it must never page with less headroom than the host reserves."""

    def test_an_off_switch_exists(self):
        tree = _tree(MEMMGR)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "maybe_enable_aimdo"
        )
        src = ast.unparse(fn)
        assert "ENABLE_ENV_VAR" in src, "the off switch is gone"
        assert "DISABLE_VALUES" in src, (
            "worker aimdo must remain defeatable without a code change"
        )

    def test_headroom_is_mirrored_not_zeroed(self):
        """A worker paging with zero headroom against a host that reserves some
        is the admission problem this seam exists to prevent. Asserted on the
        ast: the init_devices argument must reference the mirrored headroom, so
        a constant cannot sneak back in under a different spelling."""
        tree = _tree(MEMMGR)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "maybe_enable_aimdo")
        calls = [n for n in ast.walk(fn)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "init_devices"]
        assert calls, "init_devices call not found"
        for call in calls:
            names = {x.id for a in call.args for x in ast.walk(a)
                     if isinstance(x, ast.Name)}
            assert "headroom" in names, (
                "init_devices no longer passes the mirrored headroom variable."
            )


def _call_error_handlers():
    """except handlers in the worker's main() that send a full error frame."""
    tree = _tree(WORKER)
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    out = []
    for node in ast.walk(main):
        if not isinstance(node, ast.ExceptHandler):
            continue
        strings = {s.value for stmt in node.body for s in ast.walk(stmt)
                   if isinstance(s, ast.Constant) and isinstance(s.value, str)}
        if {"error", "traceback"} <= strings:
            out.append(node)
    return out


def test_a_failed_call_still_reports_the_models_it_loaded():
    """Fixed 2026-09-02: _attach_new_models runs on the error frames too. A node
    can move a 10GB model to CUDA and THEN raise (an OOM does exactly that);
    before this, that VRAM was invisible to the host for the worker's life."""
    handlers = _call_error_handlers()
    assert len(handlers) >= 2, (
        f"expected both call paths' error handlers in main(); found {len(handlers)}")
    for h in handlers:
        called = {sub.func.id for stmt in h.body for sub in ast.walk(stmt)
                  if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)}
        assert "_attach_new_models" in called, (
            f"the error frame sent from the handler at line {h.lineno} does not "
            f"call _attach_new_models."
        )


def test_worker_reserve_sums_margin_and_blindness():
    """The blindness correction (bytes siblings hold) and the host margin
    (extra_reserved_vram) serve different purposes and must ADD. max() let the
    margin vanish whenever siblings held more than it, The margin does not
    reach dynamic models (their budget is 0, rewritten to 1e32 upstream), so
    this guards the ledger-path loads only."""
    src = WORKER.read_text(encoding="utf-8")
    assert "extra_reserved = int(extra_reserved or 0) + _others" in src, (
        "the worker reserve no longer sums the host margin with the blindness "
        "correction; under any real multi-worker load the margin vanishes and "
        "near-capacity loads OOM in the cast path."
    )
    assert "extra_reserved = max(int(extra_reserved or 0), _others)" not in src


def test_host_derived_env_writes_never_clobber_pack_env_vars():
    """Fixed 2026-09-02: a pack's [env_vars] land in `env` before the
    host-derived block runs, and only ENABLE_ENV_VAR was guarded; the headroom,
    NVML and COMFY_CPU writes silently overwrote an operator's explicit pin.
    Catches: any new env[...] write in _ensure_started for a mirrored knob that
    is not guarded by a `not in env` membership test."""
    subprocess_py = SRC / "isolation" / "workers" / "subprocess.py"
    tree = _tree(subprocess_py)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_ensure_started")
    parents = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    guarded_keys = {"HEADROOM_ENV_VAR", "SIMPLE_HEADROOM_ENV_VAR",
                    "NVML_ENV_VAR", "ENABLE_ENV_VAR", "COMFY_CPU"}
    checked = 0
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Subscript)
                and isinstance(tgt.value, ast.Name) and tgt.value.id == "env"):
            continue
        key = tgt.slice
        name = (key.id if isinstance(key, ast.Name)
                else key.value if isinstance(key, ast.Constant) else None)
        if name not in guarded_keys:
            continue
        checked += 1
        cursor, guarded = node, False
        while cursor in parents:
            cursor = parents[cursor]
            if isinstance(cursor, ast.If) and "not in env" in ast.unparse(cursor.test):
                guarded = True
                break
        assert guarded, (
            f"env write for {name} at line {node.lineno} has no `not in env` "
            f"guard; it clobbers a pack's [env_vars] pin for that variable."
        )
    assert checked >= 5, (
        f"only {checked} guarded host-derived env writes found; the block was "
        f"refactored and this guard stopped seeing it."
    )


class TestPinBudgetSeam:
    """The pin split's non-negotiables, pinned at source level. The clamp is
    dark (COMFY_ENV_PIN_SPLIT=off default) but the contract must hold from
    the first commit, because flipping the default must be a one-var change."""

    def test_apply_pin_budget_is_clamp_only(self):
        """A grant may only LOWER the ceiling, never below held bytes, and a
        disabled local MAX stays disabled. Catches: a 'helpful' rewrite that
        raises MAX toward the grant, re-enabling pinning against a mirrored
        --disable-pinned-memory."""
        tree = _tree(MEMMGR)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "apply_pin_budget")
        src = ast.unparse(fn)
        assert "max(min(local, grant), held)" in src, (
            "the clamp formula changed; grants can now RAISE the ceiling")
        assert "grant > 0 and local > 0" in src, (
            "the disabled-stays-disabled gate is gone; a grant now re-enables "
            "pinning against host intent")

    def test_headroom_mirror_assigns_directly_never_via_setter(self):
        """set_ram_cache_release_state also stamps a None callback; the only
        legitimate write is direct assignment on comfy.memory_management."""
        src = MEMMGR.read_text(encoding="utf-8")
        assert "RAM_CACHE_HEADROOM = headroom" in src.replace("cm.", ""), (
            "the headroom mirror no longer assigns RAM_CACHE_HEADROOM directly")
        calls = {n.func.attr for n in ast.walk(_tree(MEMMGR))
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)}
        assert "set_ram_cache_release_state" not in calls, (
            "the mirror now goes through the setter, stamping a None callback")

    def test_grant_fields_only_under_auto_mode(self):
        """COMFY_ENV_PIN_SPLIT=off (the shipped default) must be
        byte-identical to today: no pin_max in any reply. Catches: the mode
        gate quietly dropped from the reply builder."""
        tree = _tree(POOL)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_maybe_add_pin_grant")
        src = ast.unparse(fn)
        assert "_pin_split_mode() != 'auto'" in src
        assert "pin_max" in src

    def test_reply_is_the_only_grant_channel(self):
        """Grants ride the request_vram_budget reply and nothing else: no
        parent push exists at node boundaries, and a second channel would be
        a second clock. Catches: a grant write sneaking into the census
        ingest path."""
        tree = _tree(POOL)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_pin_ingest")
        src = ast.unparse(fn)
        assert "pin_max" not in src and "apply_pin_budget" not in src

    def test_dead_worker_report_leaves_the_ledger(self):
        """_remove_worker must pop the pin report: a retained key keeps
        splitting the pool with a ghost."""
        tree = _tree(POOL)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_remove_worker")
        src = ast.unparse(fn)
        assert "_PIN_REPORTS.pop" in src and "_PIN_GRANTS.pop" in src

    def test_worker_applies_grants_from_the_budget_reply(self):
        """The worker must read pin_max/pin_headroom off the reply and route
        them through apply_pin_budget (grow before load). Catches: the reply
        fields shipping with no consumer."""
        src = WORKER.read_text(encoding="utf-8")
        assert 'result.get("pin_max")' in src
        assert 'grant=result.get("pin_max")' in src
        assert 'headroom=result.get("pin_headroom")' in src
