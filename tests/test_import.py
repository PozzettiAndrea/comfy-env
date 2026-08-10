"""Contract: `import comfy_env` works and the public surface resolves."""

import comfy_env


def test_public_surface_resolves():
    assert comfy_env.__version__
    for name in comfy_env.__all__:
        assert getattr(comfy_env, name, None) is not None, f"__all__ export missing: {name}"


def test_three_call_contract_present():
    assert callable(comfy_env.install)
    assert callable(comfy_env.setup_env)
    assert callable(comfy_env.register_nodes)
