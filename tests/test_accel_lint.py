"""Contract: the accelerator import lint flags exactly the fatal pattern."""

import textwrap

from comfy_env.lint import lint_accelerator_imports


def _env(tmp_path, name="nodes"):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "comfy-env.toml").write_text(
        '[cuda]\npackages = ["fake-cumesh"]\n', encoding="utf-8")
    return d


def test_unguarded_toplevel_import_is_error(tmp_path):
    env = _env(tmp_path)
    (env / "bad.py").write_text("import fake_cumesh\n", encoding="utf-8")
    findings = lint_accelerator_imports(tmp_path)
    assert [f["level"] for f in findings] == ["error"]
    assert "bad.py" in findings[0]["file"]


def test_guarded_toplevel_import_is_advisory(tmp_path):
    env = _env(tmp_path)
    (env / "guarded.py").write_text(textwrap.dedent("""
        try:
            import fake_cumesh
        except ImportError:
            fake_cumesh = None
    """), encoding="utf-8")
    findings = lint_accelerator_imports(tmp_path)
    assert [f["level"] for f in findings] == ["advisory"]


def test_lazy_import_in_declared_node_is_clean(tmp_path):
    env = _env(tmp_path)
    (env / "good.py").write_text(textwrap.dedent("""
        class GpuNode:
            ACCELERATOR = "cuda"
            def run(self):
                import fake_cumesh
                return fake_cumesh.VALUE
    """), encoding="utf-8")
    assert lint_accelerator_imports(tmp_path) == []


def test_torch_cuda_in_undeclared_module_is_advisory(tmp_path):
    env = _env(tmp_path)
    (env / "opportunistic.py").write_text(textwrap.dedent("""
        class MaybeGpuNode:
            def run(self):
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                return dev
    """), encoding="utf-8")
    findings = lint_accelerator_imports(tmp_path)
    assert [f["level"] for f in findings] == ["advisory"]
    assert "torch.cuda" in findings[0]["message"]
