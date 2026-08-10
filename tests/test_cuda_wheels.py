"""Contract: wheel resolution picks the right wheel for a combo, offline."""

import io

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
