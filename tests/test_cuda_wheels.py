"""Contract: wheel resolution picks the right wheel for a combo, offline."""

import io

import comfy_env.detection.arch as arch_mod
import comfy_env.packages.cuda_wheels as cw

INDEX_HTML = """
<a href="https://example.test/dl/pkg-1.9+cu128torch2.8-cp312-cp312-manylinux_2_35_x86_64.whl">pkg-1.9+cu128torch2.8-cp312-cp312-manylinux_2_35_x86_64.whl</a><br>
<a href="https://example.test/dl/pkg-1.10+cu128torch2.8-cp312-cp312-manylinux_2_35_x86_64.whl">pkg-1.10+cu128torch2.8-cp312-cp312-manylinux_2_35_x86_64.whl</a><br>
<a href="https://example.test/dl/pkg-1.10+cu128torch2.8-cp312-cp312-linux_x86_64.whl">pkg-1.10+cu128torch2.8-cp312-cp312-linux_x86_64.whl</a><br>
<a href="https://example.test/dl/pkg-1.10+cu128torch2.8-cp312-cp312-win_amd64.whl">pkg-1.10+cu128torch2.8-cp312-cp312-win_amd64.whl</a><br>
<a href="https://example.test/dl/pkg-1.10+cu124torch2.4-cp312-cp312-manylinux_2_35_x86_64.whl">pkg-1.10+cu124torch2.4-cp312-cp312-manylinux_2_35_x86_64.whl</a><br>
<a href="https://example.test/dl/pkg-1.10+cu128torch2.8-cp310-cp310-manylinux_2_35_x86_64.whl">pkg-1.10+cu128torch2.8-cp310-cp310-manylinux_2_35_x86_64.whl</a><br>
"""


def test_get_wheel_url_matches_combo_and_prefers_manylinux(monkeypatch):
    monkeypatch.setattr(cw, "_fetch_with_retries", lambda url, timeout=10, log=None: INDEX_HTML)
    monkeypatch.setattr(cw, "_platform_tags", lambda: ["manylinux", "linux"])
    url = cw.get_wheel_url("pkg", torch_version="2.8.0", cuda_version="12.8",
                           python_version="3.12")
    assert url is not None
    assert "+cu128torch2.8" in url
    assert "cp312" in url
    assert "manylinux" in url


def test_get_wheel_url_returns_none_when_no_combo(monkeypatch):
    monkeypatch.setattr(cw, "_fetch_with_retries", lambda url, timeout=10, log=None: INDEX_HTML)
    monkeypatch.setattr(cw, "_platform_tags", lambda: ["manylinux", "linux"])
    monkeypatch.setattr(cw, "_fetch_from_github_api",
                        lambda *a, **k: None)  # keep the fallback offline too
    url = cw.get_wheel_url("pkg", torch_version="2.99.0", cuda_version="13.9",
                           python_version="3.12")
    assert url is None


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_find_matching_wheel_picks_highest_version_numerically(monkeypatch):
    # 1.9 vs 1.10: plain string comparison would pick 1.9. The contract is
    # numeric ordering.
    monkeypatch.setattr(cw.urllib.request, "urlopen",
                        lambda url, timeout=10: _FakeResponse(INDEX_HTML.encode()))
    spec = cw.find_matching_wheel("pkg", torch_version="2.8.0", cuda_version="12.8")
    assert spec == "pkg===1.10+cu128torch2.8"


def test_version_key_ordering():
    assert cw._version_key("1.10") > cw._version_key("1.9")
    assert cw._version_key("0.0.1") < cw._version_key("1.0")
    # Pre-release suffixes sort under the release, PEP-440-like.
    assert cw._version_key("2.0rc1") < cw._version_key("2.0")
    assert cw._version_key("2.0") < cw._version_key("2.0.1")


# --- CPU architecture -------------------------------------------------------
# A wheel can be perfectly right about the GPU and still be unloadable because
# the CPU is the wrong kind. These pin the two places that used to assume x86.

ARM_INDEX_HTML = """
<a href="https://example.test/dl/pkg-1.0+cu128torch2.8-cp312-cp312-manylinux_2_35_x86_64.whl">pkg-1.0+cu128torch2.8-cp312-cp312-manylinux_2_35_x86_64.whl</a><br>
<a href="https://example.test/dl/pkg-1.0+cu130torch2.10-cp312-cp312-manylinux_2_39_aarch64.whl">pkg-1.0+cu130torch2.10-cp312-cp312-manylinux_2_39_aarch64.whl</a><br>
"""


def test_cpu_arch_normalizes_arm_spellings(monkeypatch):
    for machine in ("aarch64", "arm64", "AArch64"):
        monkeypatch.setattr(arch_mod.platform, "machine", lambda m=machine: m)
        assert arch_mod.cpu_arch() == "aarch64"
    for machine in ("x86_64", "AMD64"):
        monkeypatch.setattr(arch_mod.platform, "machine", lambda m=machine: m)
        assert arch_mod.cpu_arch() == "x86_64"


def test_fallback_combo_is_per_arch():
    # ARM gets its own cell entirely: (12.8, 2.8) has no aarch64 wheels, 2.8
    # cannot reach CUDA 13 at all, and 12.8/12.9 leave Thor (sm_110) with no
    # kernel image. 13.0 is the only ARM line that carries it natively.
    assert cw.resolve_fallback_combo(arch="x86_64") == ("12.8", "2.8")
    assert cw.resolve_fallback_combo(arch="aarch64") == ("13.0", "2.10")


def test_fallback_combo_rejects_unknown_arch_and_backend():
    import pytest
    with pytest.raises(ValueError, match="riscv64"):
        cw.resolve_fallback_combo(arch="riscv64")
    with pytest.raises(ValueError, match="rocm"):
        cw.resolve_fallback_combo("rocm")


def test_platform_tags_carry_the_architecture(monkeypatch):
    monkeypatch.setattr(cw.sys, "platform", "linux")
    monkeypatch.setattr(arch_mod.platform, "machine", lambda: "x86_64")
    assert cw._platform_tags() == ["_x86_64"]
    monkeypatch.setattr(arch_mod.platform, "machine", lambda: "aarch64")
    assert cw._platform_tags() == ["_aarch64"]
    monkeypatch.setattr(cw.sys, "platform", "win32")
    assert cw._platform_tags() == ["win_amd64"]


def test_arm_host_does_not_match_an_x86_wheel(monkeypatch):
    """The regression: bare "manylinux" matched x86_64 wheels on ARM, so the
    combo resolved and pip refused the wheel much later."""
    monkeypatch.setattr(cw, "_fetch_with_retries", lambda url, timeout=10, log=None: ARM_INDEX_HTML)
    monkeypatch.setattr(cw, "_fetch_from_github_api", lambda *a, **k: None)
    monkeypatch.setattr(cw.sys, "platform", "linux")
    monkeypatch.setattr(arch_mod.platform, "machine", lambda: "aarch64")

    # The x86-only combo must not resolve on ARM.
    assert cw.get_wheel_url("pkg", torch_version="2.8.0", cuda_version="12.8",
                            python_version="3.12") is None
    # The ARM combo does.
    url = cw.get_wheel_url("pkg", torch_version="2.10.0", cuda_version="13.0",
                           python_version="3.12")
    assert url is not None and "aarch64" in url


def test_x86_host_does_not_match_an_arm_wheel(monkeypatch):
    monkeypatch.setattr(cw, "_fetch_with_retries", lambda url, timeout=10, log=None: ARM_INDEX_HTML)
    monkeypatch.setattr(cw, "_fetch_from_github_api", lambda *a, **k: None)
    monkeypatch.setattr(cw.sys, "platform", "linux")
    monkeypatch.setattr(arch_mod.platform, "machine", lambda: "x86_64")

    assert cw.get_wheel_url("pkg", torch_version="2.10.0", cuda_version="13.0",
                            python_version="3.12") is None
    url = cw.get_wheel_url("pkg", torch_version="2.8.0", cuda_version="12.8",
                           python_version="3.12")
    assert url is not None and "x86_64" in url
