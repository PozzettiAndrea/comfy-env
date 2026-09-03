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


class TestResidencyPeakSeam:
    """The peak-decay fix's wiring: what bare CI cannot execute, pinned at
    source level."""

    def test_all_frame_harvests_precede_any_status_check(self):
        """An erroring worker is exactly the one whose census and overhead
        just moved; a harvest inside a success branch goes stale when it
        matters most. Generic over every piggyback key, retroactively
        protecting the previously unguarded ones too."""
        subprocess_py = SRC / "isolation" / "workers" / "subprocess.py"
        tree = _tree(subprocess_py)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_send_request")
        keys = ("_new_models", "_vram_report", "_self_state_out")
        harvest_lines = {}
        status_line = None
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and node.args
                    and isinstance(node.args[0], ast.Constant)):
                val = node.args[0].value
                if val in keys:
                    harvest_lines.setdefault(val, node.lineno)
                elif val == "status" and status_line is None:
                    status_line = node.lineno
        missing = [k for k in keys if k not in harvest_lines]
        assert not missing, f"harvest for {missing} not found in _send_request"
        if status_line is not None:
            late = [k for k, ln in harvest_lines.items() if ln > status_line]
            assert not late, (
                f"harvests {late} sit after a status check; error frames "
                f"drop them exactly when they matter")

    def test_overhead_is_computed_from_the_census_just_built(self):
        """The mixed-frame double count: overhead from frame N combined with
        a residency census from frame N+1 under-books by the model size.
        The unified _vram_report samples all three fields in ONE pass, and
        the overhead expression must reference the census list bound in that
        same pass, never a second registry walk."""
        src = WORKER.read_text(encoding="utf-8")
        assert "_vram_report" in src and '"overhead"' in src
        block = src[src.index("_census_list = None"):src.index('"_new_models"')]
        assert "_residency_census()" in block
        assert block.count("_residency_census()") == 1, (
            "a second census walk feeds the overhead; frames can now disagree")
        for field in ('"residency"', '"overhead"', '"pinned"'):
            assert field in block, (
                f"{field} left the unified report; the co-traveling fields "
                f"have split back into separate frame keys")

    def test_parent_merges_the_report_per_field_and_route_path_never_pops(self):
        """Catches: (a) subprocess.py replacing the whole stored report
        (a frame that failed one sample erases the last good value of the
        others), pinned by requiring the pure merge helper; (b) the route
        path draining the report, which would starve the boundary ingest
        that follows the same frame."""
        sub = SRC / "isolation" / "workers" / "subprocess.py"
        tree = _tree(sub)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_send_request")
        calls = [ast.unparse(n.func) for n in ast.walk(fn)
                 if isinstance(n, ast.Call)]
        assert "merge_vram_report" in calls, (
            "_send_request no longer folds _vram_report through "
            "merge_vram_report; per-field replace is the contract")
        pool_src = POOL.read_text(encoding="utf-8")
        rt = pool_src[pool_src.index("def _routed_call"):]
        rt = rt[:rt.index("\ndef ", 1)]
        assert '.get("residency")' in rt
        assert "_last_vram_report = None" not in rt, (
            "the route path pops the report; boundary ingest then never "
            "sees pinned/overhead from that frame")
        ingest = pool_src[pool_src.index("def _ingest_worker_frames"):]
        ingest = ingest[:ingest.index("\ndef ", 1)]
        assert "_last_vram_report = None" in ingest, (
            "boundary ingest must drain the report or every boundary "
            "re-applies a stale census")

    def test_death_clears_the_overhead_ledger(self):
        tree = _tree(POOL)
        for fname in ("_remove_worker", "_cleanup_stale_patchers"):
            fn = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == fname)
            assert "_OVERHEAD_REPORTS.pop" in ast.unparse(fn), (
                f"{fname} no longer pops the overhead report; a dead worker "
                f"books phantom scratch forever")

    def test_echo_sites_route_through_apply_echo(self):
        """The peak rules live in ONE pure function; a direct peak write at
        an echo site is the shipped bug (unconditional reset) sneaking back."""
        tree = _tree(PROXY)
        for fname in ("partially_load", "partially_unload", "_mark_offloaded"):
            fn = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == fname)
            src = ast.unparse(fn)
            assert "apply_echo" in src, f"{fname} no longer routes its echo"
            assert "_residency_peak" not in src, (
                f"{fname} writes the peak directly, bypassing the "
                f"admissibility rules in state_sync.apply_echo")

    def test_in_flight_decrement_follows_the_census_apply(self):
        """The ordering that closes the mid-call-echo-then-idle window: the
        flag may clear only AFTER _register_new_patchers applied the boundary
        census, in the same finally."""
        metadata_py = SRC / "isolation" / "metadata.py"
        tree = _tree(metadata_py)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_call_in_worker")
        src = ast.unparse(fn)
        assert "worker.begin_call()" in src, "the in-flight increment is gone"
        reg = src.index("_register_new_patchers(env_dir")
        dec_candidates = [i for i in range(len(src))
                          if src.startswith("worker.end_call()", i)]
        assert dec_candidates, "the in-flight decrement is gone"
        assert min(dec_candidates) > reg, (
            "the in-flight flag clears BEFORE the boundary census applies; "
            "a mid-call echo's low peak survives into idle unrepaired")

    def test_held_bytes_goes_through_the_pure_snapshot(self):
        """The admission arithmetic must stay bare-CI drivable. Catches: a
        drive-by reintroducing n_workers * CONST beside the pure call."""
        tree = _tree(POOL)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_worker_held_bytes")
        src = ast.unparse(fn)
        assert "held_from_snapshot" in src
        assert "n_workers *" not in src and "* _WORKER_FIXED_VRAM_COST" not in src


import pytest


class TestBootstrapSeam:
    """Both worker bootstraps (pin, reserve) share the same ordering contract
    and both pool env injections share the pack-[env_vars]-wins contract.
    One parametrized guard each, closing two pre-existing coverage holes (the
    pin bootstrap's position and the pool injections' clobber guard were
    previously unpinned)."""

    @staticmethod
    def _main_body(tree):
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

    @pytest.mark.parametrize("apply_name", ["apply_pin_budget",
                                            "apply_reserve_bootstrap"])
    def test_bootstrap_applies_between_comfy_import_and_ready(self, apply_name):
        """Apply before the comfy import is silently vacuous forever (the
        sys.modules lookup no-ops and no test fails); apply after the ready
        send races the first dispatched call against a worker still on the
        upstream default. Structural (executed main-body statements), never
        lexical: nested handlers import comfy above these lines and a lexical
        guard would bless the wrong thing."""
        body = self._main_body(_tree(WORKER))
        import_lines = [n.lineno for n in body
                        if isinstance(n, ast.Import)
                        and any(a.name == "comfy.model_management"
                                for a in n.names)]
        assert import_lines, "comfy.model_management import not found in main()"
        apply_lines = [n.lineno for n in body
                       if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute)
                       and n.func.attr == apply_name]
        assert apply_lines, f"{apply_name} is never called at main-body level"
        ready_lines = [n.lineno for n in body
                       if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute)
                       and n.func.attr == "send"
                       and any(isinstance(a, ast.Name) and a.id == "_ready_frame"
                               for a in n.args)]
        assert ready_lines, "ready frame send not found"
        assert min(import_lines) < min(apply_lines) < min(ready_lines), (
            f"{apply_name} must run after the comfy import and before the "
            f"ready frame; found import={min(import_lines)} "
            f"apply={min(apply_lines)} ready={min(ready_lines)}")

    def test_pool_env_injection_never_clobbers_pack_env_vars(self):
        """Every env var the pool injects at worker creation must be guarded
        by a `not in env_vars` membership test: pack [env_vars] outranks the
        host-derived value. Covers the two pin vars (previously unguarded by
        any test) and the reserve var."""
        tree = _tree(POOL)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_get_or_create_worker")
        parents = {}
        for node in ast.walk(fn):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        guarded_names = {"PIN_SHARE_ENV_VAR", "PIN_HEADROOM_ENV_VAR",
                         "RESERVE_ENV_VAR"}
        checked = set()
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            tgt = node.targets[0]
            if not (isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "env_vars"):
                continue
            key = tgt.slice
            name = key.attr if isinstance(key, ast.Attribute) else None
            if name not in guarded_names:
                continue
            checked.add(name)
            cursor, guarded = node, False
            while cursor in parents:
                cursor = parents[cursor]
                if isinstance(cursor, ast.If) and \
                        "not in" in ast.unparse(cursor.test):
                    guarded = True
                    break
            assert guarded, (
                f"pool injection of {name} at line {node.lineno} has no "
                f"not-in guard; it clobbers a pack's [env_vars] pin")
        assert checked == guarded_names, (
            f"expected injections for {guarded_names}, found {checked}; the "
            f"guard stopped seeing the block")

    def test_reserve_injection_is_not_gated_on_pin_split(self):
        """Nesting the reserve write inside the PIN_SPLIT block would make
        the fix function only under the experimental clamp mode and stay
        dead in the shipped default."""
        tree = _tree(POOL)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_get_or_create_worker")
        parents = {}
        for node in ast.walk(fn):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(fn):
            if (isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Subscript)
                    and isinstance(node.targets[0].slice, ast.Attribute)
                    and node.targets[0].slice.attr == "RESERVE_ENV_VAR"):
                cursor = node
                while cursor in parents:
                    cursor = parents[cursor]
                    if isinstance(cursor, ast.If) and \
                            "_pin_split_mode" in ast.unparse(cursor.test):
                        raise AssertionError(
                            "the reserve bootstrap is gated on PIN_SPLIT")
                return
        raise AssertionError("RESERVE_ENV_VAR injection not found")

    def test_reserve_value_comes_from_the_settled_attribute(self):
        """The injected value must be host mm.EXTRA_RESERVED_VRAM verbatim
        bytes (the same source the budget reply reads: advance equals
        settlement by construction). Recomputing from the GB float flag is
        the unit trap that injects "8" where bytes are owed."""
        tree = _tree(POOL)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_get_or_create_worker")
        for node in ast.walk(fn):
            if (isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Subscript)
                    and isinstance(node.targets[0].slice, ast.Attribute)
                    and node.targets[0].slice.attr == "RESERVE_ENV_VAR"):
                src = ast.unparse(node.value)
                assert "EXTRA_RESERVED_VRAM" in src
                assert "1024" not in src and "reserve_vram" not in src, (
                    "the reserve value is recomputed from the flag instead "
                    "of read from the settled attribute")
                return
        raise AssertionError("RESERVE_ENV_VAR injection not found")


class TestResidencyWriterSerialization:
    """The residency writer cluster fix: every writer of _residency_* fields
    serializes on the worker's leaf mutex, and detach carries a seq."""

    def test_no_bare_peak_writes_anywhere(self):
        """Extends the model_patcher-only ban to pool.py: the route thread's
        old bare `_p._residency_peak = max(...)` was a lost-update racing a
        concurrent detach (resurrecting a peak the detach just zeroed).
        state_sync owns every peak write."""
        for path in (POOL, PROXY):
            tree = _tree(path)
            init_spans = [(n.lineno, n.end_lineno) for n in ast.walk(tree)
                          if isinstance(n, ast.FunctionDef)
                          and n.name == "__init__"]

            def in_init(lineno):
                return any(lo <= lineno <= hi for lo, hi in init_spans)

            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AugAssign):
                    targets = [node.target]
                for t in targets:
                    if in_init(node.lineno):
                        continue  # priming an UNPUBLISHED object is legal
                    assert not (isinstance(t, ast.Attribute)
                                and t.attr == "_residency_peak"), (
                        f"{path.name}:{node.lineno} writes _residency_peak "
                        f"directly; route it through state_sync under the "
                        f"worker's _mem_lock")

    def test_residency_writers_run_under_the_leaf_mutex(self):
        """Every apply_residency/apply_echo/apply_peak_raise call in pool.py
        and model_patcher.py must sit inside a `with ..._mem_lock` block:
        an unlocked writer is the equal-seq clobber reopening."""
        for path in (POOL, PROXY):
            tree = _tree(path)
            parents = {}
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    parents[child] = node
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("apply_residency", "apply_echo",
                                               "apply_peak_raise")):
                    continue
                cursor, locked = node, False
                while cursor in parents:
                    cursor = parents[cursor]
                    if isinstance(cursor, ast.With) and \
                            "_mem_lock" in ast.unparse(cursor.items[0].context_expr):
                        locked = True
                        break
                assert locked, (
                    f"{path.name}:{node.lineno} calls {node.func.attr} "
                    f"outside the worker _mem_lock")

    def test_model_to_device_echoes_carry_seq(self):
        """Both ok frames (moved True AND moved False) must bump and send
        seq: without it, a census sampled before a detach wins the tie after
        it and re-inflates a correctly zeroed model, stickily."""
        src = WORKER.read_text(encoding="utf-8")
        start = src.index("def _handle_model_to_device")
        end = src.index("def _call_parent")
        body = src[start:end]
        assert body.count('"seq": _bump_seq(_mid)') >= 2, (
            "a model_to_device ok frame lost its seq; detach-to-zero can be "
            "resurrected by a stale census again")


class TestObservabilitySeam:
    """The always-on lines: each must sit OUTSIDE any debug-flag gate, or
    the user who needs it most (no flags set, one OOM) never sees it."""

    @staticmethod
    def _log_calls_outside_debug_gates(fn):
        gated = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.If) and "_DBG_" in ast.unparse(node.test):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        gated.add(id(inner))
        out = []
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and id(node) not in gated
                    and ast.unparse(node.func) == "_log"):
                out.append(ast.unparse(node))
        return out

    def test_admission_tight_line_is_not_debug_gated(self):
        fn = next(n for n in ast.walk(_tree(POOL))
                  if isinstance(n, ast.FunctionDef) and n.name == "_handle_vram_budget")
        ungated = self._log_calls_outside_debug_gates(fn)
        assert any("admission tight" in c for c in ungated), ungated
        # and it carries the fields a reader needs to reproduce the verdict
        line = next(c for c in ungated if "admission tight" in c)
        for field in ("need=", "true_free=", "in_flight=", "forward=", "excess="):
            assert field in line, f"admission line lost {field}"

    def test_pin_regression_line_is_not_debug_gated(self):
        fn = next(n for n in ast.walk(_tree(POOL))
                  if isinstance(n, ast.FunctionDef) and n.name == "_maybe_add_pin_grant")
        src = ast.unparse(fn)
        assert "pin_regression_line" in src
        assert "_log(line)" in src
        ungated = self._log_calls_outside_debug_gates(fn)
        assert any("line" in c for c in ungated)

    def test_teardown_logs_before_removing_the_worker(self):
        """Catches: the line placed after _remove_worker (a raise inside
        removal would skip it) or missing entirely."""
        meta = SRC / "isolation" / "metadata.py"
        fn = next(n for n in ast.walk(_tree(meta))
                  if isinstance(n, ast.FunctionDef) and n.name == "_call_in_worker")
        handler = next(h for t in ast.walk(fn) if isinstance(t, ast.Try)
                       for h in t.handlers
                       if h.type is not None and "ConnectionError" in ast.unparse(h.type))
        body = [ast.unparse(st) for st in handler.body]
        log_idx = next(i for i, st in enumerate(body) if "worker teardown" in st)
        rm_idx = next(i for i, st in enumerate(body) if "_remove_worker" in st)
        assert log_idx < rm_idx

    def test_overhead_warning_threshold_is_device_shaped(self):
        fn = next(n for n in ast.walk(_tree(POOL))
                  if isinstance(n, ast.FunctionDef) and n.name == "_ingest_worker_frames")
        src = ast.unparse(fn)
        assert "overhead_warn_threshold(_device_total_bytes())" in src

    def test_worker_log_lines_carry_an_identity_prefix(self):
        """Every worker appends to one shared file; an unprefixed line from
        two envs is unattributable."""
        src = WORKER.read_text(encoding="utf-8")
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "wlog")
        assert "_WLOG_PREFIX" in ast.unparse(fn)
        assert "os.getpid()" in src[:src.index("def wlog")]
