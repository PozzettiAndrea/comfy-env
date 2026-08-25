"""Contract: a worker resolves the ComfyUI root through custom_nodes/, not a link.

abspath, not resolve(): a pack behind a junction must walk up through
custom_nodes/ into the ComfyUI tree. resolve() takes it to its physical
location instead -- which may sit under a DIFFERENT checkout, whose root then
becomes this worker's base for models, input and output.

Only test_pack_outside_any_checkout_does_not_borrow_the_link_targets_root
discriminates against the old code; the other three are regression pins for
layouts that already worked.
"""

import shutil
import sys

import pytest

from comfy_env.isolation.workers.subprocess import SubprocessWorker


def _make_comfyui(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text("# comfyui\n", encoding="utf-8")
    (root / "comfy").mkdir(exist_ok=True)
    (root / "custom_nodes").mkdir(exist_ok=True)
    return root


@pytest.fixture()
def _worker_for():
    """Construct without spawning -- but __init__ still mkdtemps a worker tree."""
    made = []

    def build(working_dir):
        w = SubprocessWorker(
            python=sys.executable,
            working_dir=working_dir,
            sys_path=[],
            name="test-worker",
        )
        made.append(w)
        return w

    yield build

    for w in made:
        shutil.rmtree(w._temp_dir, ignore_errors=True)


@pytest.mark.skipif(sys.platform == "win32", reason="needs symlink without admin")
def test_symlinked_pack_resolves_through_custom_nodes(tmp_path, _worker_for):
    """The real install must win, even when the pack physically lives elsewhere."""
    real = _make_comfyui(tmp_path / "real_comfy")

    physical = tmp_path / "elsewhere" / "MyPack"
    (physical / "nodes").mkdir(parents=True)

    link = real / "custom_nodes" / "MyPack"
    link.symlink_to(physical, target_is_directory=True)

    worker = _worker_for(link / "nodes")
    assert worker._find_comfyui_base() == real


@pytest.mark.skipif(sys.platform == "win32", reason="needs symlink without admin")
def test_pack_outside_any_checkout_does_not_borrow_the_link_targets_root(tmp_path, _worker_for):
    """The case where the deleted fallback was actively wrong, not merely useless.

    The correct walk (abspath, through custom_nodes/) must fail here -- the pack
    is not under any ComfyUI tree. The deleted fallback used .resolve(), which
    followed the link INTO a different checkout and returned that checkout as
    this worker's base. Everything the worker then did with folder_paths --
    models, input, output -- pointed at the wrong install.

    Correct answer is None: "I could not determine a base", not a guess.
    """
    decoy = _make_comfyui(tmp_path / "decoy_comfy")
    physical = decoy / "custom_nodes" / "MyPack"
    (physical / "nodes").mkdir(parents=True)

    # The pack is reached from a directory under no ComfyUI root at all.
    loose = tmp_path / "loose"
    loose.mkdir()
    link = loose / "MyPack"
    link.symlink_to(physical, target_is_directory=True)

    base = _worker_for(link / "nodes")._find_comfyui_base()

    assert base != decoy, (
        "resolved through the symlink into an unrelated checkout -- "
        "this is what the .resolve() fallback did"
    )
    assert base is None


def test_plain_pack_still_resolves(tmp_path, _worker_for):
    """No regression for the ordinary, un-linked layout."""
    real = _make_comfyui(tmp_path / "comfy")
    pack = real / "custom_nodes" / "PlainPack" / "nodes"
    pack.mkdir(parents=True)

    assert _worker_for(pack)._find_comfyui_base() == real


def test_no_comfyui_anywhere_returns_none(tmp_path, _worker_for):
    """A walk that finds nothing must terminate and return None, not guess."""
    orphan = tmp_path / "no_comfy" / "pack" / "nodes"
    orphan.mkdir(parents=True)

    assert _worker_for(orphan)._find_comfyui_base() is None
