"""Contract: pixi is pinned, verified, and never installed unverified."""

import hashlib
import io
import tarfile
import zipfile

import pytest

import comfy_env.packages.pixi as px


def test_version_is_pinned_not_latest():
    # Positive assertions only (a negative source-grep false-positived on
    # docstrings twice): the download URL is built from the pinned version.
    assert px.PIXI_VERSION != "latest"
    import inspect
    src = inspect.getsource(px.ensure_pixi)
    assert "releases/download/" in src and "v{PIXI_VERSION}" in src


def test_all_platforms_have_vendored_hashes():
    for key, (asset, sha) in px._ASSETS.items():
        assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), key
        assert asset.endswith((".tar.gz", ".zip")), key


def test_owned_path_is_version_keyed_and_not_user_pixi():
    # Version-keyed comfy-env-owned dir: bumping PIXI_VERSION re-provisions;
    # the user's own ~/.pixi install is never touched.
    assert px.PIXI_VERSION in px.PIXI
    assert ".pixi" not in px.PIXI.replace(".pixi.exe", "")
    assert ".comfy-env" in px.PIXI


def test_checksum_mismatch_refuses_install(monkeypatch, tmp_path):
    monkeypatch.setattr(px, "PIXI", str(tmp_path / "bin" / "pixi"))

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(px.urllib.request, "urlopen",
                        lambda url, context=None: FakeResp(b"malicious bytes"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        px.ensure_pixi()
    assert not (tmp_path / "bin" / "pixi").exists(), "unverified binary was written"


def test_good_checksum_installs_and_marks(monkeypatch, tmp_path):
    # Build a fake archive whose hash we vendor for this test.
    payload = b"#!/fake pixi binary"
    if px.sys.platform == "win32":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("pixi.exe", payload)
        asset_name = "fake.zip"
    else:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            info = tarfile.TarInfo("pixi")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
        asset_name = "fake.tar.gz"
    data = buf.getvalue()

    dest = tmp_path / "bin" / ("pixi.exe" if px.sys.platform == "win32" else "pixi")
    monkeypatch.setattr(px, "PIXI", str(dest))
    key = (px.platform.system(), px.platform.machine())
    monkeypatch.setitem(px._ASSETS, key, (asset_name, hashlib.sha256(data).hexdigest()))

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(px.urllib.request, "urlopen",
                        lambda url, context=None: FakeResp(data))

    assert px.ensure_pixi() == str(dest)
    assert dest.read_bytes() == payload
    # Second call short-circuits (no download): break urlopen to prove it.
    monkeypatch.setattr(px.urllib.request, "urlopen",
                        lambda url, context=None: (_ for _ in ()).throw(AssertionError("downloaded again")))
    assert px.ensure_pixi() == str(dest)
