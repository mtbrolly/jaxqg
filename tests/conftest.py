"""
Shared fixtures and configuration for jaxqg tests.
"""

import jax.numpy as jnp
import pytest

from jaxqg import Grid, Model, define_state
from jaxqg.backend import x64_enabled
from jaxqg.mechanisms import QGDynamics, SpectralFilter
from jaxqg.timesteppers import AB3
from jaxqg.utils import gaussian_random_field


@pytest.fixture
def single_precision():
    """Whether tests are running with the configured float32 backend."""
    return not x64_enabled()


@pytest.fixture
def grid_2layer():
    """2-layer grid fixture."""
    return Grid(n_x=64, n_y=64, L_x=1e6, L_y=1e6, H=[500.0, 2000.0])


@pytest.fixture
def grid_3layer():
    """3-layer grid fixture."""
    return Grid(n_x=64, n_y=64, L_x=1e6, L_y=1e6, H=[300.0, 700.0, 2000.0])


@pytest.fixture
def grid_8layer():
    """8-layer grid fixture (for multi-GPU tests)."""
    H = [500.0] + [1500.0 / 7] * 7
    return Grid(n_x=32, n_y=32, L_x=1e6, L_y=1e6, H=H)


@pytest.fixture
def model_2layer(grid_2layer):
    """2-layer model fixture."""
    timestepper = AB3(dt=1800)
    U = jnp.array([0.025, 0.0])
    mechanisms_ = {
        "QG": QGDynamics(grid_2layer, U=U, beta=1e-11, gamma=7e-8),
        "filter": SpectralFilter(grid_2layer),
    }
    return Model(grid_2layer, mechanisms_, timestepper)


@pytest.fixture
def model_3layer(grid_3layer):
    """3-layer model fixture."""
    timestepper = AB3(dt=1800)
    U = jnp.array([0.03, 0.015, 0.0])
    mechanisms_ = {
        "QG": QGDynamics(grid_3layer, U=U, beta=1e-11, gamma=7e-8),
        "filter": SpectralFilter(grid_3layer),
    }
    return Model(grid_3layer, mechanisms_, timestepper)


@pytest.fixture
def state_2layer(model_2layer):
    """Initial state for 2-layer model."""
    qh = gaussian_random_field(model_2layer, 1e-26 * model_2layer.grid.wv**-4, seed=42)
    return define_state(model_2layer, qh)


@pytest.fixture
def state_3layer(model_3layer):
    """Initial state for 3-layer model."""
    qh = gaussian_random_field(model_3layer, 1e-26 * model_3layer.grid.wv**-4, seed=42)
    return define_state(model_3layer, qh)
