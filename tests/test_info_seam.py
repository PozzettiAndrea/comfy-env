"""Contract: `comfy-env info --json` is the SUPPORTED seam for other repos.

comfy-test consumes the workspace root and the ABI tag. Before this existed it
imported `comfy_env.environment.cache.get_workspace_dir` and the private
`_abi_tag` across a repo boundary -- private names are free to move, so that
coupling broke on every refactor. These tests pin the schema, not the values.
"""

import json
import subprocess
import sys


from comfy_env.environment import RuntimeEnv
from conftest import subprocess_env

# Renaming any of these is a breaking change for comfy-test.
CONTRACT_FIELDS = {
    "os_name", "platform_tag", "cpu_arch", "python_version", "torch_version",
    "cuda_version", "gpu_name", "gpu_compute", "gpu_vram_mb",
    "workspace_dir", "abi_tag", "comfy_env_version",
}


def test_detect_populates_the_contract_fields():
    d = RuntimeEnv.detect().as_dict()
    assert set(d) == CONTRACT_FIELDS
    # These three are machine-independent and must never be None.
    assert d["os_name"] and d["platform_tag"] and d["cpu_arch"]


def test_detect_survives_a_machine_with_no_workspace_and_no_torch(monkeypatch):
    """A diagnostic that dies on an unconfigured machine is worthless."""
    import comfy_env.environment.cache as cache
    monkeypatch.setattr(cache, "get_workspace_dir",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    env = RuntimeEnv.detect()
    assert env.workspace_dir is None and env.abi_tag is None
    assert env.os_name  # the rest still resolved


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "comfy_env.cli", *args],
        capture_output=True, text=True, env=subprocess_env(),
    )


def test_info_json_stdout_is_pure_json():
    """Logs go to stderr; stdout must be machine-parseable on its own, or
    every consumer needs a fragile filter."""
    r = _cli("info", "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)          # raises if anything else leaked to stdout
    assert set(d) == CONTRACT_FIELDS


def test_info_and_doctor_actually_run():
    """Both were dead for nine days behind an import of a deleted symbol,
    with no test to notice. This is that test."""
    for args in (("info",), ("doctor",)):
        r = _cli(*args)
        assert r.returncode == 0, f"{args}: {r.stderr}"
        assert "cannot import name" not in r.stderr
