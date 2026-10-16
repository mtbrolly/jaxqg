"""
Tests for jaxqg.grid module.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxqg import Grid, Model, define_state
from jaxqg.backend import COMPLEX
from jaxqg.mechanisms import QGDynamics, SpectralFilter
from jaxqg.timesteppers import AB3
from jaxqg.utils import gaussian_random_field


class TestGridInitialization:
    """Test Grid initialisation and basic properties."""

    def test_2layer_grid(self, grid_2layer):
        """Test 2-layer grid creation."""
        assert grid_2layer.n_layers == 2
        assert grid_2layer.n_x == 64
        assert grid_2layer.n_y == 64
        assert grid_2layer.n_ky == 64
        assert grid_2layer.n_kx == 33  # n_x // 2 + 1

    @pytest.mark.parametrize(
        ("n_x", "n_y", "dimension"),
        [(7, 8, "n_x"), (8, 7, "n_y")],
    )
    def test_odd_grid_dimensions_are_rejected(self, n_x, n_y, dimension):
        with pytest.raises(ValueError, match=f"{dimension} must be even"):
            Grid(n_x=n_x, n_y=n_y)

    def test_single_layer_barotropic_grid(self):
        """A barotropic grid has no vertical interfaces to index."""
        grid = Grid(n_x=8, H=[1000.0], r_d=1e10)

        assert grid.n_layers == 1
        assert grid.F.shape == (1,)
        assert grid.F_interface_up.shape == (0,)
        assert grid.F_interface_down.shape == (0,)
        assert jnp.all(jnp.isfinite(grid.A))

    def test_3layer_grid(self, grid_3layer):
        """Test 3-layer grid creation."""
        assert grid_3layer.n_layers == 3
        assert len(grid_3layer.H) == 3
        assert len(grid_3layer.F) == 3

    def test_grid_dimensions(self, grid_2layer):
        """Test grid spatial dimensions."""
        assert grid_2layer.x.shape == (64, 64)
        assert grid_2layer.y.shape == (64, 64)
        assert grid_2layer.kx.shape == (64, 33)
        assert grid_2layer.ky.shape == (64, 33)

    def test_grid_wavenumbers(self, grid_2layer):
        """Test wavenumber properties."""
        assert grid_2layer.wv2.shape == (64, 33)
        assert jnp.all(grid_2layer.wv2 >= 0)
        assert grid_2layer.wv.shape == (64, 33)


class TestGridMatrices:
    """Test inversion matrices."""

    def test_inversion_matrix_shape(self, grid_2layer):
        """Test A matrix shape for inversion."""
        assert grid_2layer.A.shape == (2, 2, 64, 33)

    def test_inversion_matrix_exists(self, grid_2layer):
        """Test that A matrix is computed."""
        assert grid_2layer.A is not None
        assert not jnp.any(jnp.isnan(grid_2layer.A))

    def test_ph_from_qh(self, grid_2layer):
        """Test streamfunction inversion from PV."""
        qh = jnp.ones((2, 64, 33), dtype=COMPLEX)
        ph = grid_2layer._ph_from_qh(qh)
        assert ph.shape == qh.shape
        assert ph.dtype == qh.dtype

    def test_qh_from_ph(self, grid_2layer):
        """Test PV from streamfunction."""
        ph = jnp.ones((2, 64, 33), dtype=COMPLEX)
        qh = grid_2layer._qh_from_ph(ph)
        assert qh.shape == ph.shape
        assert qh.dtype == ph.dtype


class TestStretchingParameters:
    """Test stretching parameter calculations."""

    def test_F_vectors_2layer(self, grid_2layer):
        """Test stretching parameters for 2-layer model."""
        assert len(grid_2layer.F) == 2
        assert grid_2layer.F[0] > 0
        assert grid_2layer.F[1] > 0

    def test_F_vectors_3layer(self, grid_3layer):
        """Test stretching parameters for 3-layer model."""
        assert len(grid_3layer.F) == 3
        assert jnp.all(grid_3layer.F > 0)

    def test_interface_F_parameters(self, grid_2layer):
        """Test interface stretching parameters."""
        assert len(grid_2layer.F_interface_up) == 1
        assert len(grid_2layer.F_interface_down) == 1
        assert grid_2layer.F_interface_up[0] > 0
        assert grid_2layer.F_interface_down[0] > 0


class TestGridSpectralProperties:
    """Test spectral space properties."""

    def test_wavenumber_zero_mode(self, grid_2layer):
        """Test zero wavenumber properties."""
        # Zero wavenumber should be at (0, 0) in frequency space
        assert grid_2layer.kx[0, 0] == 0.0
        assert grid_2layer.ky[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_wavenumber_symmetry(self, grid_2layer):
        """Test wavenumber grid properties."""
        # Real FFT gives positive kx only
        assert jnp.all(grid_2layer.kx >= 0)
        # ky should be symmetric
        assert jnp.any(grid_2layer.ky > 0)
        assert jnp.any(grid_2layer.ky < 0)

    def test_deformation_radius(self):
        """Test deformation radius parameter."""
        grid = Grid(n_x=32, r_d=15000.0)
        assert grid.wv_d == pytest.approx(1.0 / 15000.0)


class TestLazyB:
    """Test lazy (on-demand) construction of the B matrix."""

    def test_B_not_materialized_at_init(self, grid_3layer):
        """B is dead memory in the integration path; not built at init."""
        assert grid_3layer._B_cache is None

    def test_B_cached_after_access(self, grid_3layer):
        B = grid_3layer.B
        assert grid_3layer._B_cache is not None
        assert grid_3layer.B is B

    def test_B_values_match_reference(self, grid_3layer):
        """B matches an explicitly constructed reference."""
        g = grid_3layer
        B_ref = np.zeros((g.n_layers, g.n_layers, g.n_ky, g.n_kx), dtype=np.complex128)
        wv2 = np.asarray(g.wv2)
        F = np.asarray(g.F)
        F_up = np.asarray(g.F_interface_up)
        F_dn = np.asarray(g.F_interface_down)
        for i in range(g.n_layers):
            B_ref[i, i] = -wv2 - F[i]
            if i < g.n_layers - 1:
                B_ref[i, i + 1] = F_up[i]
            if i > 0:
                B_ref[i, i - 1] = F_dn[i - 1]
        assert jnp.allclose(g.B, B_ref)

    def test_qh_from_ph_uses_lazy_B(self, grid_2layer):
        """_qh_from_ph still works through the lazy property."""
        ph = jnp.ones((2, 64, 33), dtype=COMPLEX)
        qh = grid_2layer._qh_from_ph(ph)
        assert qh.shape == ph.shape
        assert qh.dtype == ph.dtype

    def test_B_setter_back_compatible(self, grid_2layer):
        """Assigning grid.B still works (populates the cache)."""
        B_old = grid_2layer.B
        grid_2layer.B = 2 * B_old
        assert jnp.allclose(grid_2layer.B, 2 * B_old)

    def test_physical_first_radius_can_be_requested(self):
        """Grid calibrates its coupling scale from the physical target."""
        from jaxqg.diagnostics import vertical_modes

        target = 15000.0
        grid = Grid(n_x=32, L_x=1e6, L_y=1e6, H=[125.0] * 4 + [1250.0] * 4, r_d=target)

        class DummyModel:
            pass

        model = DummyModel()
        model.grid = grid
        _, radii = vertical_modes(model)
        assert float(radii[1]) == pytest.approx(target, rel=1e-10)
        assert grid.wv_d != pytest.approx(1.0 / target)


multi_device = pytest.mark.skipif(
    jax.device_count() < 2,
    reason="requires at least 2 local devices "
    "(or XLA_FLAGS=--xla_force_host_platform_device_count=N)",
)


class TestShardedA:
    """Test layer-sharded construction of the A inversion matrix."""

    def test_single_device_list_is_dense(self):
        """A single device in a_shard_devices keeps the dense behaviour."""
        grid_dense = Grid(n_x=32, L_x=1e6, L_y=1e6, H=[500.0, 1000.0, 1500.0, 2000.0])
        grid_sharded = Grid(
            n_x=32,
            L_x=1e6,
            L_y=1e6,
            H=[500.0, 1000.0, 1500.0, 2000.0],
            a_shard_devices=jax.devices()[:1],
        )
        assert grid_sharded._a_state_sharding is None
        assert jnp.allclose(grid_dense.A, grid_sharded.A)

    def test_aux_data_preserves_shard_spec(self):
        """Pytree round-trip preserves the a_shard_devices spec."""
        devices = jax.devices()[:1]
        grid = Grid(n_x=32, H=[500.0, 5000.0], a_shard_devices=devices)
        _, aux_data = Grid.tree_flatten(grid)
        assert aux_data["a_shard_devices"] == devices
        grid2 = Grid.tree_unflatten(aux_data, ())
        assert grid2._a_shard_devices == devices

    def test_nondivisible_devices_fall_back_to_dense(self):
        """n_layers not divisible by device count warns and uses dense A."""
        grid_dense = Grid(n_x=32, L_x=1e6, L_y=1e6, H=[500.0, 1000.0, 1500.0])
        grid_sharded = Grid(
            n_x=32,
            L_x=1e6,
            L_y=1e6,
            H=[500.0, 1000.0, 1500.0],
            a_shard_devices=jax.devices(),
        )
        assert grid_sharded._a_state_sharding is None
        assert jnp.allclose(grid_dense.A, grid_sharded.A)

    @multi_device
    def test_sharded_A_matches_dense(self):
        """Sharded-A equals dense-A (N > 2 generic inverse)."""
        H = [500.0, 750.0, 1250.0, 3000.0]
        grid_dense = Grid(n_x=32, L_x=1e6, L_y=1e6, H=H)
        grid_sharded = Grid(
            n_x=32, L_x=1e6, L_y=1e6, H=H, a_shard_devices=jax.devices()[:2]
        )
        A_sharded = np.asarray(grid_sharded.A)
        assert A_sharded.shape == np.asarray(grid_dense.A).shape
        assert jnp.allclose(grid_sharded.A, grid_dense.A, rtol=1e-10, atol=1e-10)

    @multi_device
    def test_sharded_A_2layer_matches_dense(self):
        """The 2-layer analytic path respects the same sharding."""
        grid_dense = Grid(n_x=32, L_x=1e6, L_y=1e6, H=[500.0, 5000.0])
        grid_sharded = Grid(
            n_x=32,
            L_x=1e6,
            L_y=1e6,
            H=[500.0, 5000.0],
            a_shard_devices=jax.devices()[:2],
        )
        assert jnp.allclose(grid_sharded.A, grid_dense.A, rtol=1e-10, atol=1e-10)

    @multi_device
    def test_sharded_integration_matches_unsharded(self):
        """Full integration trajectory equivalence (small grid, same seed)."""
        H = [500.0, 750.0, 1250.0, 3000.0]
        U = jnp.array([0.025, 0.015, 0.005, 0.0])

        grid_dense = Grid(n_x=32, L_x=1e6, L_y=1e6, H=H)
        model_dense = Model(
            grid_dense,
            {
                "QG": QGDynamics(grid_dense, U=U, beta=1e-11, gamma=7e-8),
                "filter": SpectralFilter(grid_dense),
            },
            AB3(1800.0),
        )

        grid_sharded = Grid(
            n_x=32, L_x=1e6, L_y=1e6, H=H, a_shard_devices=jax.devices()[:2]
        )
        model_sharded = Model(
            grid_sharded,
            {
                "QG": QGDynamics(grid_sharded, U=U, beta=1e-11, gamma=7e-8),
                "filter": SpectralFilter(grid_sharded),
            },
            AB3(1800.0),
        )

        qh = gaussian_random_field(model_dense, 1e-26 * grid_dense.wv**-4, seed=42)

        traj_dense = model_dense.generate_integration_fn(n_steps=8, sample_rate=4)(
            define_state(model_dense, qh)
        )

        from jaxqg import shard_state_across_layers

        state_sharded = shard_state_across_layers(
            define_state(model_sharded, qh), devices=jax.devices()[:2]
        )
        traj_sharded = model_sharded.generate_integration_fn(n_steps=8, sample_rate=4)(
            state_sharded
        )

        scale = float(jnp.abs(traj_dense.qh).max())
        assert jnp.allclose(
            traj_sharded.qh, traj_dense.qh, rtol=1e-6, atol=1e-8 * scale
        )
        assert not jnp.any(jnp.isnan(traj_sharded.qh))
