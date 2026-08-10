"""Pixi binary management: pinned version, checksum-verified, comfy-env-owned.

The pinned pixi lives in a comfy-env-owned, VERSION-KEYED directory
(~/.comfy-env/pixi/<version>/). The path existing IS the version check --
no marker files -- and bumping PIXI_VERSION (plus hashes) provisions the
new version on every machine without ever touching a user's own pixi
install at ~/.pixi. The download is the official release archive for the
pinned version, verified against a sha256 vendored here from the release's
sha256.sum; a checksum mismatch refuses to install.
"""

import hashlib
import io
import os
import platform
import ssl
import stat
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

PIXI_VERSION = "0.75.0"

_name = "pixi.exe" if sys.platform == "win32" else "pixi"
# comfy-env-owned install root -- deliberately NOT ~/.pixi, which belongs
# to the user's own pixi installation and must never be clobbered.
PIXI_HOME = Path.home() / ".comfy-env" / "pixi" / PIXI_VERSION
PIXI = str(PIXI_HOME / _name)

# (asset archive name, sha256) per platform -- hashes from the official
# sha256.sum of the pinned release. The bare binaries are not individually
# hashed upstream; the archives are, so we download and extract those.
_ASSETS = {
    ("Linux", "x86_64"): (
        "pixi-x86_64-unknown-linux-musl.tar.gz",
        "bcd825d62905c29b3c754b71f9cdc9d6f119454398f58330f111a7b6a0de0a3f"),
    ("Linux", "aarch64"): (
        "pixi-aarch64-unknown-linux-musl.tar.gz",
        "6476588859faa7232def49ff590f199390bd93fae3826617596befc52382724f"),
    ("Darwin", "x86_64"): (
        "pixi-x86_64-apple-darwin.tar.gz",
        "f129e890366ad5502304c8f863cc5585d82143b64731f36bcd1283a27781097e"),
    ("Darwin", "arm64"): (
        "pixi-aarch64-apple-darwin.tar.gz",
        "52a43f9268f3accb7155cf229937f2f5333559b1333615d610362c6151fded66"),
    ("Windows", "AMD64"): (
        "pixi-x86_64-pc-windows-msvc.zip",
        "0c478f9efcb0f8ba984b21c3fa9f484a3d098fc9a46be896f0ac260939ddeaa9"),
}


def _extract_binary(asset_name: str, data: bytes) -> bytes:
    """Pull the pixi binary out of the release archive."""
    if asset_name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            member = next(n for n in zf.namelist() if n.endswith(_name))
            return zf.read(member)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("pixi"))
        f = tf.extractfile(member)
        assert f is not None
        return f.read()


def ensure_pixi():
    """Ensure the pinned pixi version is installed at the comfy-env-owned
    path. Downloads the pinned release archive, verifies its vendored
    sha256, extracts the binary. Raises RuntimeError on checksum mismatch.
    """
    if Path(PIXI).exists():
        return PIXI

    key = (platform.system(), platform.machine())
    asset = _ASSETS.get(key)
    if not asset:
        raise RuntimeError(f"No pixi binary for {key[0]}/{key[1]}")
    asset_name, expected_sha = asset
    url = (f"https://github.com/prefix-dev/pixi/releases/download/"
           f"v{PIXI_VERSION}/{asset_name}")

    print(f"[comfy-env] installing pixi {PIXI_VERSION}...", file=sys.stderr, flush=True)

    # Portable/embedded Python often lacks CA certs; use certifi if available
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, context=ctx) as resp:
        data = resp.read()

    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"pixi download checksum mismatch for {asset_name}: "
            f"expected {expected_sha}, got {actual_sha}. "
            f"Refusing to install an unverified binary.")

    binary = _extract_binary(asset_name, data)
    dest = Path(PIXI)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(binary)
    if sys.platform != "win32":
        tmp.chmod(tmp.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.replace(tmp, dest)

    print(f"[comfy-env] pixi {PIXI_VERSION} installed: {PIXI}", file=sys.stderr, flush=True)
    return PIXI
