"""Contract: stale comfy-env pins in sibling packs are surfaced, not silent."""

import comfy_env
from comfy_env.install.plugin import check_sibling_comfy_env_pins


def _pack(custom_nodes, name, requirements=None):
    d = custom_nodes / name
    d.mkdir(parents=True)
    if requirements is not None:
        (d / "requirements.txt").write_text(requirements, encoding="utf-8")
    return d


def test_stale_pins_are_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy_env, "__version__", "0.4.12")
    custom_nodes = tmp_path / "custom_nodes"
    me = _pack(custom_nodes, "ComfyUI-Mine", "numpy\ncomfy-env==0.3.9\n")
    _pack(custom_nodes, "ComfyUI-Stale", "torch\ncomfy-env==0.3.9\n")
    _pack(custom_nodes, "ComfyUI-Capped", "comfy_env<=0.4.0  # old cap\n")
    _pack(custom_nodes, "ComfyUI-Fine", "comfy-env>=0.4.0\nnumpy\n")
    _pack(custom_nodes, "ComfyUI-NoReq")

    messages = []
    findings = check_sibling_comfy_env_pins(me, log=messages.append)

    flagged = {name for name, _ in findings}
    # Stale exact pin and stale upper bound flagged; >= is fine; the calling
    # pack itself is excluded (its own requirements are reinstalled anyway).
    assert flagged == {"ComfyUI-Stale", "ComfyUI-Capped"}
    assert all("DOWNGRADED" in m for m in messages)


def test_current_and_newer_pins_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy_env, "__version__", "0.4.12")
    custom_nodes = tmp_path / "custom_nodes"
    me = _pack(custom_nodes, "ComfyUI-Mine")
    _pack(custom_nodes, "ComfyUI-Exact", "comfy-env==0.4.12\n")
    _pack(custom_nodes, "ComfyUI-Newer", "comfy-env==0.5.0\n")
    _pack(custom_nodes, "ComfyUI-Unpinned", "comfy-env\n")

    assert check_sibling_comfy_env_pins(me, log=lambda m: None) == []
