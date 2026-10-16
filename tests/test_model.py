"""
Tests for jaxqg.model module.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxqg import Model, define_state
from jaxqg.backend import COMPLEX
from jaxqg.mechanisms import QGDynamics, SpectralFilter
from jaxqg.timesteppers import AB3


class TestStateDefinition:
    """Test model state initialisation."""

    def test_state_creation_2layer(self, model_2layer, state_2layer):
        """Test state creation for 2-layer model."""
        assert state_2layer.qh.shape == (2, 64, 33)
        assert state_2layer.tn == 0
        assert state_2layer.random_key is not None

    def test_state_creation_3layer(self, model_3layer, state_3layer):
        """Test state creation for 3-layer model."""
        assert state_3layer.qh.shape == (3, 64, 33)
        assert state_3layer.tn == 0

    def test_state_rhs_initialization(self, state_2layer):
        """Test that RHS fields are initialised."""
        assert state_2layer.rhs_m1.shape == state_2layer.qh.shape
        assert state_2layer.rhs_m2.shape == state_2layer.qh.shape
        assert jnp.allclose(state_2layer.rhs_m1, 0)
        assert jnp.allclose(state_2layer.rhs_m2, 0)

    def test_state_wrong_shape_raises(self, model_2layer):
        """Test that wrong qh shape raises error."""
        qh_wrong = jnp.ones((3, 64, 33), dtype=COMPLEX)  # Wrong n_layers
        with pytest.raises(ValueError):
            define_state(model_2layer, qh_wrong)


class TestModelIntegration:
    """Test model time integration."""

    @pytest.mark.parametrize(
        "mechanism_names", [(), ("QG",), ("filter",), ("spectral_filter",)]
    )
    def test_optional_mechanisms(self, model_2layer, state_2layer, mechanism_names):
        """Either mechanism can be omitted from an experiment."""
        available_mechanisms = {
            "QG": model_2layer.mechanisms["QG"],
            "filter": model_2layer.mechanisms["filter"],
            "spectral_filter": model_2layer.mechanisms["filter"],
        }
        mechanisms = {
            name: available_mechanisms[name] for name in mechanism_names
        }
        model = Model(model_2layer.grid, mechanisms, model_2layer.timestepper)

        new_state = model.evolve_one_step_stateless(None, state_2layer)

        assert new_state.tn == state_2layer.tn + 1
        if mechanism_names in (("filter",), ("spectral_filter",)):
            assert jnp.allclose(
                new_state.qh, available_mechanisms[mechanism_names[0]](state_2layer.qh)
            )
        elif not mechanism_names:
            assert jnp.allclose(new_state.qh, state_2layer.qh)

    def test_single_step(self, model_2layer, state_2layer):
        """Test a single time step."""
        qh_initial = state_2layer.qh.copy()
        new_state = model_2layer.evolve_one_step_stateless(None, state_2layer)

        assert new_state.tn == 1
        assert new_state.qh.shape == qh_initial.shape
        assert not jnp.allclose(new_state.qh, qh_initial)  # Should evolve

    def test_multi_step_integration(self, model_2layer, state_2layer):
        """Test multiple time steps."""
        integrate = model_2layer.generate_integration_fn(n_steps=10, sample_rate=5)
        traj = integrate(state_2layer)

        # Should have 2 output states (n_steps // sample_rate)
        assert traj.qh.shape[0] == 2
        assert traj.qh.shape[1:] == state_2layer.qh.shape

    def test_integration_energy_dissipation(self, model_2layer, state_2layer):
        """Test that energy decreases with dissipation."""
        from jaxqg import EKE

        eke_initial = EKE(model_2layer, state_2layer.qh)

        integrate = model_2layer.generate_integration_fn(n_steps=10, sample_rate=1)
        traj = integrate(state_2layer)

        eke_final = EKE(model_2layer, traj.qh[-1])

        # With diffusion, energy should decrease
        assert eke_final < eke_initial

    def test_3layer_integration(self, model_3layer, state_3layer):
        """Test integration with 3 layers."""
        integrate = model_3layer.generate_integration_fn(n_steps=5, sample_rate=5)
        traj = integrate(state_3layer)

        assert traj.qh.shape == (1, 3, 64, 33)
        assert not jnp.any(jnp.isnan(traj.qh))


class TestModelJIT:
    """Test JIT compilation of model."""

    def test_step_jit(self, model_2layer, state_2layer):
        """Test JIT compiled time stepping."""
        step_jit = jax.jit(lambda s: model_2layer.evolve_one_step_stateless(None, s))
        new_state = step_jit(state_2layer)

        assert new_state.tn == 1
        assert new_state.qh.shape == state_2layer.qh.shape

    def test_integration_jit(self, model_2layer, state_2layer):
        """Test JIT compiled integration."""
        integrate_fn = model_2layer.generate_integration_fn(n_steps=5, sample_rate=5)
        integrate_jit = jax.jit(integrate_fn)

        traj = integrate_jit(state_2layer)
        assert traj.qh.shape[0] == 1


class TestStateNumericalStability:
    """Test numerical stability."""

    def test_no_nan_generation(self, model_2layer, state_2layer):
        """Test that integration doesn't generate NaNs."""
        integrate = model_2layer.generate_integration_fn(n_steps=20, sample_rate=20)
        traj = integrate(state_2layer)

        assert not jnp.any(jnp.isnan(traj.qh))
        assert not jnp.any(jnp.isinf(traj.qh))

    def test_small_timestep_convergence(self, model_2layer, state_2layer):
        """Test that smaller timesteps converge towards the finer solution."""
        grid = model_2layer.grid
        U = jnp.array([0.025, 0.0])

        def make_model(dt):
            mechanisms_ = {
                "QG": QGDynamics(grid, U=U, beta=1e-11, gamma=7e-8),
                "filter": SpectralFilter(grid),
            }
            return Model(grid, mechanisms_, AB3(dt))

        model_dt = model_2layer
        model_half = make_model(1800 / 2)
        model_quarter = make_model(1800 / 4)

        integrate = model_dt.generate_integration_fn(n_steps=10, sample_rate=10)
        traj_dt = integrate(state_2layer)
        assert traj_dt.qh.shape == (1, 2, 64, 33)

        integrate_half = model_half.generate_integration_fn(n_steps=20, sample_rate=20)
        traj_half = integrate_half(define_state(model_half, state_2layer.qh))
        integrate_quarter = model_quarter.generate_integration_fn(
            n_steps=40, sample_rate=40
        )
        traj_quarter = integrate_quarter(define_state(model_quarter, state_2layer.qh))

        finest = traj_quarter.qh[0]
        err = float(jnp.max(jnp.abs(traj_dt.qh[0] - finest)))
        err2 = float(jnp.max(jnp.abs(traj_half.qh[0] - finest)))

        # The per-step SpectralFilter is dt-independent (applied once per step),
        # so its O(dt) accumulated effect dominates AB3's O(dt^3) truncation and
        # the gap to the finest solution shrinks by only ~1.9-2.1x (not ~8x)
        # when dt is halved. Assert the gap strictly decreases.
        assert err2 < err


class TestModelSerialization:
    """Test model serialisation properties."""

    def test_state_is_namedtuple(self, state_2layer):
        """Test that State is a proper NamedTuple."""
        assert hasattr(state_2layer, "qh")
        assert hasattr(state_2layer, "tn")
        assert hasattr(state_2layer, "random_key")
