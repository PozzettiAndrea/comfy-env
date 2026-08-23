"""Contract: the root role schema -- comfy-env-root.toml carries [node_reqs]
and [types] only; anything else is rejected at parse time, generically
(no legacy keys are named anywhere in code)."""

import pytest

from comfy_env.config import load_config


def _root(tmp_path, body):
    p = tmp_path / "comfy-env-root.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_root_rejects_unsupported_sections(tmp_path):
    p = _root(tmp_path, '[env_vars]\nFOO = "1"\n\n[cuda]\npackages = ["x"]\n')
    # Assert on the offending section names, not the prose -- the tail of this
    # message is documentation and changes (it gained [types] in 0.4.23).
    with pytest.raises(ValueError, match=r"unsupported section\(s\) \[cuda\], \[env_vars\]"):
        load_config(p)


def test_root_rejects_typos(tmp_path):
    # A typo'd section is an error, not a silent no-op.
    with pytest.raises(ValueError, match=r"\[setings\]"):
        load_config(_root(tmp_path, "[setings]\npool_ipc = true\n"))


def test_root_accepts_its_schema(tmp_path):
    cfg = load_config(_root(
        tmp_path,
        '[node_reqs]\nOtherPack = "https://github.com/x/OtherPack"\n\n'
        '[types]\nWIDGET = "builtin"\n'))
    assert cfg.node_reqs[0]["name"] == "OtherPack"
    assert cfg.types == {"WIDGET": "builtin"}


def test_empty_root_is_fine(tmp_path):
    assert load_config(_root(tmp_path, "# empty marker\n")) is not None


def test_env_file_is_not_role_checked(tmp_path):
    # The env role keeps its open schema (pixi passthrough).
    p = tmp_path / "comfy-env.toml"
    p.write_text('[env_vars]\nFOO = "1"\n\n[dependencies]\nffmpeg = "*"\n',
                 encoding="utf-8")
    cfg = load_config(p)
    assert cfg.env_vars == {"FOO": "1"}
    assert cfg.pixi_passthrough["dependencies"] == {"ffmpeg": "*"}


def test_install_fails_on_bad_root_config(tmp_path, monkeypatch):
    pack = tmp_path / "custom_nodes" / "ComfyUI-X"
    pack.mkdir(parents=True)
    (pack / "comfy-env-root.toml").write_text('[env_vars]\nFOO = "1"\n', encoding="utf-8")

    from comfy_env.install import install
    with pytest.raises(ValueError, match="unsupported section"):
        install(node_dir=pack, log_callback=lambda m: None, dry_run=True)
