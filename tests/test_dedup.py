"""Contract: identical packages across envs share physical storage and RAM.

Disk: pixi (conda side) and uv (pypi side) install by HARDLINKING from a
central cache -- N envs with the same resolved package cost one copy on
disk. This only helps when envs resolve the SAME version+build: envs pinned
to different torch combos legitimately share nothing.

RAM: persistent workers that map the same torch files share the read-only
code pages; each extra worker costs roughly its private portion, not a full
torch. This too depends on the disk-level sharing (same file => same pages).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.network
def test_pixi_envs_hardlink_identical_packages(tmp_path):
    """Two envs, same tiny conda python + pypi package: files must share
    inodes (one physical copy via the pixi/uv caches)."""
    from comfy_env.detection import get_pixi_platform
    from comfy_env.packages.pixi import PIXI, ensure_pixi

    ensure_pixi()
    roots = []
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        (d / "pixi.toml").write_text(
            f'[workspace]\nname = "dedup-{name}"\n'
            f'channels = ["conda-forge"]\nplatforms = ["{get_pixi_platform()}"]\n\n'
            '[dependencies]\npython = "3.11.*"\n\n'
            '[pypi-dependencies]\nsix = "*"\n',
            encoding="utf-8")
        subprocess.run(
            [str(PIXI), "install", "--manifest-path", str(d / "pixi.toml")],
            check=True, capture_output=True, text=True, timeout=900)
        roots.append(d / ".pixi" / "envs" / "default")

    def stat_of(root: Path, filename: str):
        hit = next(p for p in root.rglob(filename) if p.is_file())
        return os.stat(hit)

    # pypi side (installed by uv under pixi)
    sa, sb = stat_of(roots[0], "six.py"), stat_of(roots[1], "six.py")
    assert sa.st_nlink >= 2, "env file not hardlinked to any cache"
    assert (sa.st_ino, sa.st_dev) == (sb.st_ino, sb.st_dev), \
        "identical pypi package files are separate physical copies across envs"

    # conda side (linked by pixi from the rattler cache)
    exe = "python.exe" if sys.platform == "win32" else "libpython3.11.so.1.0"
    ca, cb = stat_of(roots[0], exe), stat_of(roots[1], exe)
    assert (ca.st_ino, ca.st_dev) == (cb.st_ino, cb.st_dev), \
        "identical conda package files are separate physical copies across envs"


def test_workers_share_torch_code_pages():
    """Two persistent workers mapping the SAME torch files must share
    read-only pages: rss - uss (private) per worker must show real sharing.
    Guards against a regression to copy-per-worker loading."""
    psutil = pytest.importorskip("psutil")
    from comfy_env.isolation.workers.subprocess import SubprocessWorker

    workers, pids = [], []
    try:
        for i in range(2):
            w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                                 name=f"dedup-{i}")
            # The Popen pid can be a wrapper (sandboxed shells); ask the
            # worker itself.
            pids.append(w.call_module(module="os", func="getpid"))
            w.call_module(module="echo_node", func="make_tensor",
                          rows=256, cols=256)
            workers.append(w)

        shared = []
        for pid in pids:
            mi = psutil.Process(pid).memory_full_info()
            assert mi.uss < mi.rss, "uss >= rss: measurement is broken"
            shared.append(mi.rss - mi.uss)
        # Both workers map the same torch binaries -> tens of MB of shared
        # code pages each. 10 MB is a deliberately loose floor.
        assert min(shared) > 10 * 1024 * 1024, \
            f"workers show almost no page sharing: {[s // 1024 // 1024 for s in shared]} MB"
    finally:
        for w in workers:
            w.shutdown()
