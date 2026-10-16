"""
Tests for jaxqg.mechanisms module.
"""

import jax
import jax.numpy as jnp

from jaxqg.backend import COMPLEX
from jaxqg.mechanisms import QGDynamics, SpectralFilter


class TestQGDynamics:
    """Test QG dynamics mechanisms."""

    def test_qg_initialization_2layer(self, grid_2layer):
        """Test QG dynamics initialisation for 2 layers."""
        dynamics = QGDynamics(grid_2layer, beta=1e-11, gamma=7e-8)
        assert dynamics.grid == grid_2layer
        assert dynamics.beta == 1e-11
        assert dynamics.gamma == 7e-8

    def test_qg_mean_flow(self, grid_2layer):
        """Test mean flow parameters."""
        U = jnp.array([0.025, 0.0])
        dynamics = QGDynamics(grid_2layer, U=U, beta=1e-11)
        assert jnp.allclose(dynamics.U, U)
        assert dynamics.U.shape == (2,)

    def test_qg_beta_vectors(self, grid_2layer):
        """Test beta vector initialisation."""
        dynamics = QGDynamics(grid_2layer, beta=1e-11)
        assert dynamics.beta_vec.shape == (2, 64, 33)
        assert not jnp.any(jnp.isnan(dynamics.beta_vec))

    def test_qg_rhs_computation(self, grid_2layer):
        """Test RHS computation."""
        dynamics = QGDynamics(grid_2layer, beta=1e-11)
        qh = jnp.ones((2, 64, 33), dtype=COMPLEX)
        rhs = dynamics._rhs(qh)

        assert rhs.shape == qh.shape
        assert rhs.dtype == qh.dtype
        assert not jnp.any(jnp.isnan(rhs))

    def test_qg_bottom_drag(self, grid_2layer):
        """Test bottom drag application."""
        # With gamma > 0, bottom layer should be affected
        dynamics_with_drag = QGDynamics(grid_2layer, gamma=1e-6)
        dynamics_no_drag = QGDynamics(grid_2layer, gamma=0.0)

        qh = jnp.ones((2, 64, 33), dtype=COMPLEX) * (1 + 1j)
        rhs_drag = dynamics_with_drag._rhs(qh)
        rhs_no_drag = dynamics_no_drag._rhs(qh)

        # Results should differ (bottom layer affected by drag)
        assert not jnp.allclose(rhs_drag, rhs_no_drag)

    def test_mean_flow_effect(self, grid_2layer):
        """Test effect of mean flow on dynamics."""
        U_shear = jnp.array([0.1, 0.0])
        U_no_shear = jnp.array([0.0, 0.0])

        dyn_shear = QGDynamics(grid_2layer, U=U_shear, beta=1e-11)
        dyn_no_shear = QGDynamics(grid_2layer, U=U_no_shear, beta=1e-11)

        qh = jnp.ones((2, 64, 33), dtype=COMPLEX)
        rhs_shear = dyn_shear._rhs(qh)
        rhs_no_shear = dyn_no_shear._rhs(qh)

        # Mean flow should change RHS
        assert not jnp.allclose(rhs_shear, rhs_no_shear)


class TestSpectralFilter:
    """Test spectral filtering."""

    def test_filter_initialization(self, grid_2layer):
        """Test spectral filter initialisation."""
        filt = SpectralFilter(grid_2layer)
        assert filt.exp_filter.shape == (64, 33)
        assert jnp.all((filt.exp_filter >= 0) & (filt.exp_filter <= 1))

    def test_filter_application(self, grid_2layer):
        """Test filter application to field."""
        filt = SpectralFilter(grid_2layer)
        qh = jnp.ones((2, 64, 33), dtype=COMPLEX)
        filtered = filt(qh)

        assert filtered.shape == qh.shape
        # Filter should reduce magnitude
        assert jnp.abs(filtered).max() <= jnp.abs(qh).max() + 1e-10

    def test_filter_multi_layer(self, grid_3layer):
        """Test filter on multi-layer field."""
        filt = SpectralFilter(grid_3layer)
        qh = jnp.ones((3, 64, 33), dtype=COMPLEX)
        filtered = filt(qh)

        assert filtered.shape == (3, 64, 33)
        # Filter should be same across layers
        assert jnp.allclose(filtered[0], filtered[1])
        assert jnp.allclose(filtered[1], filtered[2])

    def test_filter_preserves_low_wavenumbers(self, grid_2layer):
        """Test that filter preserves low wavenumber modes."""
        filt = SpectralFilter(grid_2layer)
        # Low frequencies should have filter value close to 1
        assert filt.exp_filter[0, 0] > 0.99


class TestDynamicsConsistency:
    """Test consistency of dynamics calculations."""

    def test_jax_jit_consistency(self, grid_2layer):
        """Test that JIT compilation preserves results."""
        dynamics = QGDynamics(grid_2layer, beta=1e-11)
        qh = jnp.ones((2, 64, 33), dtype=COMPLEX)

        # Compute without JIT (via method call)
        rhs1 = dynamics._rhs(qh)

        # Compute with JIT
        jitted_rhs = jax.jit(dynamics._rhs)
        rhs2 = jitted_rhs(qh)

        assert jnp.allclose(rhs1, rhs2)

    def test_3layer_consistency(self, grid_3layer):
        """Test 3-layer dynamics consistency."""
        dynamics = QGDynamics(grid_3layer, beta=1e-11)
        qh = jnp.ones((3, 64, 33), dtype=COMPLEX)
        rhs = dynamics._rhs(qh)

        assert rhs.shape == (3, 64, 33)
        assert not jnp.any(jnp.isnan(rhs))
        assert not jnp.any(jnp.isinf(rhs))


class TestLazyBroadcastConstants:
    """Test the lazily broadcast mean-flow / PV-gradient constants."""

    def test_constants_not_materialized_at_init(self, grid_2layer):
        """The (n_layers, ny, nx)-sized broadcasts are not built at init."""
        dynamics = QGDynamics(grid_2layer, beta=1e-11)
        assert dynamics._beta_vec_cache is None
        assert dynamics._Qy_vec_cache is None
        assert dynamics._U_vec_cache is None
        assert dynamics._V_vec_cache is None
        # (n_layers,) scalars are available immediately
        assert dynamics.beta_layer.shape == (2,)
        assert dynamics.Qy_layer.shape == (2,)

    def test_lazy_vec_properties(self, grid_2layer):
        """Back-compatible properties materialize with correct shapes."""
        U = jnp.array([0.025, 0.0])
        V = jnp.array([0.01, -0.01])
        dynamics = QGDynamics(grid_2layer, U=U, V=V, beta=1e-11)

        assert dynamics.beta_vec.shape == (2, 64, 33)
        assert dynamics.Qy_vec.shape == (2, 64, 33)
        assert dynamics.U_vec.shape == (2, 64, 64)
        assert dynamics.V_vec.shape == (2, 64, 64)

        assert jnp.allclose(dynamics.beta_vec[0], dynamics.beta_layer[0])
        assert jnp.allclose(dynamics.U_vec[1], U[1])
        assert jnp.allclose(dynamics.V_vec[1], V[1])

    def test_rhs_matches_materialized_reference(self, grid_2layer):
        """_rhs (lazy broadcasting) equals a reference using explicit
        materialised broadcasts (regression test for the refactoring)."""
        U = jnp.array([0.025, 0.0])
        V = jnp.array([0.01, -0.01])
        beta = 1e-11
        gamma = 7e-8
        dynamics = QGDynamics(grid_2layer, U=U, V=V, beta=beta, gamma=gamma)

        qh = (
            jnp.asarray(
                jax.random.normal(jax.random.PRNGKey(1), (2, 64, 33)), dtype=COMPLEX
            )
            * 1e-20
        )
        rhs = dynamics._rhs(qh)

        # Reference with fully materialised constants
        U_vec = jnp.broadcast_to(U[:, None, None], (2,) + grid_2layer.x.shape)
        V_vec = jnp.broadcast_to(V[:, None, None], (2,) + grid_2layer.x.shape)
        beta_vec = jnp.broadcast_to(dynamics.beta_layer[:, None, None], (2, 64, 33))
        Qy_vec = jnp.broadcast_to(dynamics.Qy_layer[:, None, None], (2, 64, 33))

        ph = grid_2layer._ph_from_qh(qh)
        uh = -grid_2layer.iky * ph
        vh = grid_2layer.ikx * ph
        q = jnp.fft.irfft2(qh)
        u = jnp.fft.irfft2(uh)
        v = jnp.fft.irfft2(vh)
        uq = (U_vec + u) * q
        vq = (V_vec + v) * q
        uqh = jnp.fft.rfft2(uq)
        vqh = jnp.fft.rfft2(vq)
        advection = -(grid_2layer.ikx * uqh + grid_2layer.iky * vqh)
        beta_term = -beta_vec * grid_2layer.ikx * ph
        meridional_term = Qy_vec * grid_2layer.iky * ph
        rhs_ref = advection + beta_term + meridional_term
        bf = gamma * grid_2layer.wv2 * ph
        bf = bf.at[:-1].set(0.0)
        rhs_ref = rhs_ref + bf

        assert jnp.allclose(rhs, rhs_ref, rtol=1e-12, atol=1e-30)

    def test_shear_contributes_to_beta_layer(self, grid_2layer):
        """Vertical shear enters the effective beta (same as before)."""
        U_shear = jnp.array([0.025, 0.0])
        dyn = QGDynamics(grid_2layer, U=U_shear, beta=1e-11)
        g = grid_2layer
        # beta_layer[0] = beta + F_down[0]*(U_0-U_1)
        expected = 1e-11 + g.F_interface_down[0] * (0.025 - 0.0)
        assert jnp.allclose(dyn.beta_layer[0], expected)
        assert jnp.allclose(dyn.beta_layer[1], 1e-11)

    def test_uniform_meridional_flow_advects_pv(self, grid_2layer):
        """Uniform V contributes the expected -V * d(q)/dy advection."""
        V0 = 0.03
        phase = 2 * jnp.pi * grid_2layer.y / grid_2layer.L_y
        q_real = jnp.cos(phase)
        qh = jnp.fft.rfft2(jnp.broadcast_to(q_real, (2,) + q_real.shape)).astype(
            COMPLEX
        )
        dynamics_zero = QGDynamics(grid_2layer)
        dynamics_uniform = QGDynamics(grid_2layer, V=[V0, V0])

        rhs_delta = dynamics_uniform._rhs(qh) - dynamics_zero._rhs(qh)
        expected = -V0 * grid_2layer.iky * qh

        assert jnp.allclose(rhs_delta[:, 1, 0], expected[:, 1, 0], rtol=2e-7, atol=1e-12)

    def test_meridional_shear_terms_match_pv_gradient(self, grid_2layer):
        """V shear produces the correct Qy gradient and linear RHS term."""
        V = jnp.array([0.01, -0.02])
        dynamics_zero = QGDynamics(grid_2layer)
        dynamics_shear = QGDynamics(grid_2layer, V=V)
        expected_Qy = jnp.array(
            [
                grid_2layer.F_interface_up[0] * (V[1] - V[0]),
                grid_2layer.F_interface_down[0] * (V[0] - V[1]),
            ]
        )
        phase = 2 * jnp.pi * grid_2layer.y / grid_2layer.L_y
        q_real = jnp.cos(phase)
        qh = jnp.fft.rfft2(jnp.broadcast_to(q_real, (2,) + q_real.shape)).astype(
            COMPLEX
        )

        assert jnp.allclose(dynamics_shear.Qy_layer, expected_Qy)

        ph = grid_2layer._ph_from_qh(qh)
        expected_delta = (
            -V[:, None, None] * grid_2layer.iky * qh
            + expected_Qy[:, None, None] * grid_2layer.iky * ph
        )
        rhs_delta = dynamics_shear._rhs(qh) - dynamics_zero._rhs(qh)
        assert jnp.allclose(
            rhs_delta[:, 1, 0], expected_delta[:, 1, 0], rtol=2e-7, atol=1e-12
        )
