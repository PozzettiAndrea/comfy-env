"""Contract: inert root-config sections warn instead of silently no-oping."""


def test_root_inert_sections_warn(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_ENV_INSTALL_ISOLATED", "0")  # skip workspace half
    pack = tmp_path / "custom_nodes" / "ComfyUI-X"
    pack.mkdir(parents=True)
    (pack / "comfy-env-root.toml").write_text(
        '[env_vars]\nFOO = "1"\n\n[cuda]\npackages = ["cumesh"]\n',
        encoding="utf-8")

    from comfy_env.install import install
    messages = []
    assert install(node_dir=pack, log_callback=messages.append, dry_run=True)
    joined = "\n".join(messages)
    assert "[env_vars] in comfy-env-root.toml has no effect" in joined
    assert "[cuda] in comfy-env-root.toml has no effect" in joined


def test_clean_root_config_does_not_warn(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_ENV_INSTALL_ISOLATED", "0")
    pack = tmp_path / "custom_nodes" / "ComfyUI-Y"
    pack.mkdir(parents=True)
    (pack / "comfy-env-root.toml").write_text(
        "[settings]\nisolate = true\n", encoding="utf-8")

    from comfy_env.install import install
    messages = []
    assert install(node_dir=pack, log_callback=messages.append, dry_run=True)
    assert "has no effect" not in "\n".join(messages)
