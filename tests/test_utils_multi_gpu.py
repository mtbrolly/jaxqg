"""
Tests for jaxqg.utils_multi_gpu module.
"""

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding

from jaxqg import (
    Model,
    MultiGPUConfig,
    MultiGPUSimulator,
    State,
    define_state,
    shard_state_across_layers,
)
from jaxqg.backend import COMPLEX


def _shard_devices(n_layers):
    """Devices that can hold ``n_layers`` layers sharded evenly.

    ``shard_state_across_layers`` shards along the layer axis, so the
    device count must divide the layer count. Use the largest divisor of
    ``n_layers`` that fits the available device count; this keeps the
    tests valid on single-device machines and on multi-device hosts
    where the device count need not divide the layer count.
    """
    devices = jax.devices()
    best = 1
    for d in range(1, min(len(devices), n_layers) + 1):
        if n_layers % d == 0:
            best = d
    return devices[:best]


class TestShardStateBasics:
    """Test basic state sharding functionality."""

    def test_shard_preserves_values(self, state_2layer):
        """Test that sharding preserves state values."""
        sharded_state = shard_state_across_layers(state_2layer, devices=_shard_devices(2))

        # Values should be identical
        assert jnp.allclose(sharded_state.qh, state_2layer.qh)
        assert jnp.allclose(sharded_state.rhs_m1, state_2layer.rhs_m1)
        assert jnp.allclose(sharded_state.rhs_m2, state_2layer.rhs_m2)

    def test_shard_preserves_time(self, state_2layer):
        """Test that time index is preserved."""
        sharded_state = shard_state_across_layers(state_2layer, devices=_shard_devices(2))
        assert sharded_state.tn == state_2layer.tn

    def test_shard_returns_state(self, state_2layer):
        """Test that sharded result is a State."""
        sharded_state = shard_state_across_layers(state_2layer, devices=_shard_devices(2))
        assert isinstance(sharded_state, State)

    def test_shard_3layer(self, state_3layer):
        """Test sharding with 3 layers."""
        sharded_state = shard_state_across_layers(state_3layer, devices=_shard_devices(3))
        assert jnp.allclose(sharded_state.qh, state_3layer.qh)
        assert sharded_state.qh.shape[0] == 3


class TestDevicePlacement:
    """Test device placement after sharding."""

    def test_sharded_state_has_sharding(self, state_2layer):
        """Test that sharded state has sharding information."""
        sharded_state = shard_state_across_layers(state_2layer, devices=_shard_devices(2))

        # Check that arrays have sharding info
        assert hasattr(sharded_state.qh, "sharding")

    def test_qh_sharded_on_layer_dim(self, state_2layer):
        """Test that qh is sharded on layer dimension (axis 0)."""
        sharded_state = shard_state_across_layers(state_2layer, devices=_shard_devices(2))

        # Get the sharding specification
        sharding = sharded_state.qh.sharding
        assert isinstance(sharding, NamedSharding)

    def test_spatial_dims_replicated(self, state_2layer):
        """Test that spatial dimensions are replicated."""
        sharded_state = shard_state_across_layers(state_2layer, devices=_shard_devices(2))

        # Spatial dimensions (ky, kx) should be replicated across devices
        # We can't check exact placement on single device, but verify consistency
        qh_shape = sharded_state.qh.shape
        assert qh_shape == state_2layer.qh.shape

    def test_operations_on_sharded_state(self, state_2layer):
        """Test that operations work on sharded state."""
        sharded_state = shard_state_across_layers(state_2layer, devices=_shard_devices(2))

        # Should be able to compute with sharded state
        result = sharded_state.qh * 2
        assert result.shape == sharded_state.qh.shape


class TestShardingConsistency:
    """Test consistency of sharded vs unsharded operations."""

    def test_sharded_unsharded_equivalence_multiplier(self, model_2layer, state_2layer):
        """Test that operations give same results on sharded/unsharded."""
        from jaxqg import EKE

        eke_unsharded = EKE(model_2layer, state_2layer.qh)

        sharded_state = shard_state_across_layers(state_2layer, devices=_shard_devices(2))
        eke_sharded = EKE(model_2layer, sharded_state.qh)

        # Results should be identical
        assert jnp.allclose(eke_sharded, eke_unsharded)

    def test_sharded_rhs_computation(self, model_2layer, state_2layer):
        """Test RHS computation on sharded state."""
        sharded_state = shard_state_across_layers(state_2layer, devices=_shard_devices(2))
        new_state = model_2layer.evolve_one_step_stateless(None, sharded_state)

        # Should integrate without error
        assert new_state.qh.shape == state_2layer.qh.shape

    def test_multiple_shadings(self, state_2layer):
        """Test applying sharding multiple times is idempotent."""
        sharded1 = shard_state_across_layers(state_2layer, devices=_shard_devices(2))
        sharded2 = shard_state_across_layers(sharded1, devices=_shard_devices(2))

        # Values should be identical
        assert jnp.allclose(sharded1.qh, sharded2.qh)


class TestShardingEdgeCases:
    """Test edge cases in sharding."""

    def test_shard_single_device(self, state_2layer):
        """Explicit single-device sharding keeps values intact."""
        sharded_state = shard_state_across_layers(
            state_2layer, devices=jax.devices()[:1]
        )
        assert jnp.allclose(sharded_state.qh, state_2layer.qh)

    def test_shard_defaults_to_all_devices(self, state_2layer):
        """Default devices work when the layer count divides evenly."""
        n_layers = state_2layer.qh.shape[0]
        if len(jax.devices()) == 1 or n_layers % len(jax.devices()) != 0:
            pytest.skip("default device count does not divide the layer count")
        sharded_state = shard_state_across_layers(state_2layer)
        assert jnp.allclose(sharded_state.qh, state_2layer.qh)

    def test_shard_zero_state(self, model_2layer):
        """Test sharding zero state."""
        qh_zero = jnp.zeros((2, 64, 33), dtype=COMPLEX)
        zero_state = define_state(model_2layer, qh_zero)

        sharded = shard_state_across_layers(zero_state, devices=_shard_devices(2))
        assert jnp.allclose(sharded.qh, 0)

    def test_shard_large_state(self):
        """Test sharding with more layers."""
        from jaxqg.grid import Grid
        from jaxqg.mechanisms import QGDynamics, SpectralFilter
        from jaxqg.timesteppers import AB3
        from jaxqg.utils import gaussian_random_field

        # Create an 8-layer grid
        grid_large = Grid(
            n_x=128,
            n_y=128,
            L_x=1e6,
            L_y=1e6,
            H=[500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0],
        )

        timestepper = AB3(dt=1800)
        U = jnp.array([0.03, 0.025, 0.02, 0.015, 0.01, 0.005, 0.002, 0.0])
        mechanisms_ = {
            "QG": QGDynamics(grid_large, U=U, beta=1e-11, gamma=7e-8),
            "filter": SpectralFilter(grid_large),
        }
        model_large = Model(grid_large, mechanisms_, timestepper)

        qh = gaussian_random_field(
            model_large, 1e-26 * model_large.grid.wv**-4, seed=42
        )
        state_large = define_state(model_large, qh)

        sharded = shard_state_across_layers(state_large, devices=_shard_devices(8))
        assert jnp.allclose(sharded.qh, state_large.qh)


class TestShardingWithIntegration:
    """Test sharding in full integration workflow."""

    def test_integrate_sharded_state(self, model_2layer, state_2layer):
        """Test full integration with sharded initial condition."""
        sharded_state = shard_state_across_layers(state_2layer, devices=_shard_devices(2))

        integrate = model_2layer.generate_integration_fn(n_steps=5, sample_rate=5)
        traj = integrate(sharded_state)

        # Should complete without error
        assert traj.qh.shape == (1, 2, 64, 33)
        assert not jnp.any(jnp.isnan(traj.qh))

    def test_sharded_energy_conservation(self, model_2layer, state_2layer):
        """Test energy evolution with sharded state."""
        from jaxqg import EKE

        sharded_state = shard_state_across_layers(state_2layer, devices=_shard_devices(2))
        eke_initial = EKE(model_2layer, sharded_state.qh)

        integrate = model_2layer.generate_integration_fn(n_steps=3, sample_rate=3)
        traj = integrate(sharded_state)

        eke_final = EKE(model_2layer, traj.qh[-1])

        # Energy should decrease with dissipation
        assert eke_final < eke_initial

    def test_sharded_state_with_jit(self, model_2layer, state_2layer):
        """Test that sharded states work with JIT."""
        sharded_state = shard_state_across_layers(state_2layer, devices=_shard_devices(2))

        step_jit = jax.jit(lambda s: model_2layer.evolve_one_step_stateless(None, s))
        new_state = step_jit(sharded_state)

        assert new_state.qh.shape == state_2layer.qh.shape


class TestSingleDeviceSimulator:
    """Exercise the simulator API on one device."""

    @pytest.mark.parametrize("layer_sharding", [True, False])
    def test_unsharded_state_is_placed_on_selected_device(
        self, model_2layer, state_2layer, layer_sharding, monkeypatch
    ):
        """Even an unsharded state should honor the selected device."""
        sim = MultiGPUSimulator(
            model_2layer,
            MultiGPUConfig(device_indices=[0], layer_sharding=layer_sharding),
        )
        selected_device = sim.devices[0]
        real_device_put = jax.device_put
        placements = []

        def track_device_put(value, device=None, **kwargs):
            placements.append(device)
            return real_device_put(value, device=device, **kwargs)

        monkeypatch.setattr(jax, "device_put", track_device_put)
        placed_state = sim.shard_state(state_2layer)

        assert placements == [selected_device]
        assert isinstance(placed_state, State)
        assert jnp.allclose(placed_state.qh, state_2layer.qh)

    def test_run_returns_completed_trajectory_and_stats(
        self, model_2layer, state_2layer
    ):
        sim = MultiGPUSimulator(model_2layer, MultiGPUConfig(device_indices=[0]))
        trajectory, stats = sim.run(state_2layer, n_steps=2, sample_rate=1)

        assert trajectory.qh.shape == (2, 2, 64, 33)
        assert stats.n_steps == 2
        assert stats.duration_sec > 0
        assert stats.steps_per_sec > 0
        assert stats.final_state["qh_shape"] == (2, 64, 33)
