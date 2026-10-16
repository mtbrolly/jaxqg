"""Backend policy must be established in a fresh process."""

import os
import subprocess
import sys

import pytest


def _run_backend_probe(extra_env, code):
    env = os.environ.copy()
    env.pop("JAX_ENABLE_X64", None)
    env.pop("JAX_PLATFORMS", None)
    env.pop("JAX_PLATFORM_NAME", None)
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("env", "expected_float", "expected_complex"),
    [
        ({"JAX_PLATFORMS": "cpu"}, "float64", "complex128"),
        ({"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "0"}, "float32", "complex64"),
        ({"JAX_ENABLE_X64": "0"}, "float32", "complex64"),
        ({"JAX_PLATFORMS": "mps", "JAX_ENABLE_X64": "0"}, "float32", "complex64"),
    ],
)
def test_precision_policy_in_fresh_process(env, expected_float, expected_complex):
    x64 = expected_float == "float64"
    code = f"""
import jax
import jax.numpy as jnp
import jaxqg
from jaxqg import backend
assert jax.config.jax_enable_x64 is {x64}
assert backend.FLOAT == jnp.{expected_float}
assert backend.COMPLEX == jnp.{expected_complex}
if backend.selected_platform() != "metal":
    grid = jaxqg.Grid(8, H=[1., 2., 3.])
    assert grid.H.dtype == jnp.{expected_float}
    assert grid.A.dtype == jnp.{expected_complex}
"""
    result = _run_backend_probe(env, code)
    assert result.returncode == 0, result.stderr


def test_default_metal_backend_uses_float32_without_platform_environment():
    code = """
import jax
import jax.numpy as jnp
jax.default_backend = lambda: "METAL"
import jaxqg
from jaxqg import backend
assert not jax.config.jax_enable_x64
assert backend.FLOAT == jnp.float32
assert backend.COMPLEX == jnp.complex64
"""
    result = _run_backend_probe({}, code)
    assert result.returncode == 0, result.stderr


def test_precision_policy_uses_active_default_backend():
    code = """
import jax
platform = jax.default_backend().lower()
import jaxqg
expected_x64 = platform not in {"metal", "mps"}
assert jax.config.jax_enable_x64 is expected_x64
"""
    result = _run_backend_probe({}, code)
    assert result.returncode == 0, result.stderr


def test_metal_rejects_x64_override_before_backend_initialization():
    result = _run_backend_probe(
        {"JAX_PLATFORMS": "mps", "JAX_ENABLE_X64": "1"},
        "import jaxqg",
    )
    assert result.returncode != 0
    assert "Metal backend does not support 64-bit arrays" in result.stderr


def test_explicit_device_order_and_invalid_requests(monkeypatch):
    from jaxqg import backend

    devices = [object(), object(), object()]
    monkeypatch.setattr(backend.jax, "devices", lambda: devices)
    assert backend.select_devices([2, 0], prefer_gpu=False) == [devices[2], devices[0]]
    with pytest.raises(ValueError, match="Unavailable device indices"):
        backend.select_devices([3], prefer_gpu=False)
    with pytest.raises(ValueError, match="duplicates"):
        backend.select_devices([1, 1], prefer_gpu=False)
    with pytest.raises(ValueError, match="at least one"):
        backend.select_devices([], prefer_gpu=False)
