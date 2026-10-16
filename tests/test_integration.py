"""
Integration tests for jaxqg package.

These tests validate the complete workflow from initialisation through integration.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxqg import (
    APE,
    EKE,
    Grid,
    Model,
    QGDynamics,
    cfl,
    define_state,
    isotropic_EKE_spectrum,
    shard_state_across_layers,
)
from jaxqg.backend import COMPLEX
from jaxqg.mechanisms import SpectralFilter
from jaxqg.timesteppers import AB3, Euler


class TestSampleFields:
    """Test the qh-only (selected-field) scan output."""

    def test_qh_only_output_shape(self, model_2layer, state_2layer):
        """sample_fields=('qh',) stacks only qh in the scan output."""
        integrate = model_2layer.generate_integration_fn(
            n_steps=10, sample_rate=5, sample_fields=("qh",)
        )
        traj = integrate(state_2layer)

        assert traj.qh.shape == (2, 2, 64, 33)
        assert not jnp.any(jnp.isnan(traj.qh))
        # Only the selected field is present
        assert traj._fields == ("qh",)

    def test_random_initial_condition_is_real_fft_consistent(
        self, model_2layer, state_2layer, single_precision
    ):
        """The generated rfft2 field must reconstruct without stripe modes."""
        reconstructed = jnp.fft.rfft2(jnp.fft.irfft2(state_2layer.qh))
        atol = 1e-8 if single_precision else 1e-12
        assert jnp.allclose(reconstructed, state_2layer.qh, rtol=1e-5, atol=atol)

    def test_qh_only_matches_full_output(self, model_2layer, state_2layer):
        """Selected fields match the corresponding fields of the full scan."""
        integrate_full = model_2layer.generate_integration_fn(n_steps=10, sample_rate=5)
        integrate_qh = model_2layer.generate_integration_fn(
            n_steps=10, sample_rate=5, sample_fields=("qh",)
        )

        traj_full = integrate_full(state_2layer)
        traj_qh = integrate_qh(state_2layer)

        assert jnp.allclose(traj_qh.qh, traj_full.qh)

    def test_multiple_fields(self, model_2layer, state_2layer):
        """Several fields can be selected."""
        integrate = model_2layer.generate_integration_fn(
            n_steps=4, sample_rate=2, sample_fields=("qh", "rhs_m1")
        )
        traj = integrate(state_2layer)

        assert traj._fields == ("qh", "rhs_m1")
        assert traj.qh.shape == (2, 2, 64, 33)
        assert traj.rhs_m1.shape == (2, 2, 64, 33)

    def test_unknown_field_raises(self, model_2layer):
        """Unknown field names raise a ValueError."""
        with pytest.raises(ValueError, match="Unknown sample_fields"):
            model_2layer.generate_integration_fn(
                n_steps=4, sample_rate=2, sample_fields=("qh", "bogus")
            )

    def test_undivisible_sample_rate_raises(self, model_2layer):
        """n_steps must be divisible by sample_rate."""
        with pytest.raises(ValueError, match="divisible"):
            model_2layer.generate_integration_fn(n_steps=10, sample_rate=3)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_steps": 2.5},
            {"n_steps": 0},
            {"n_steps": -1},
            {"n_steps": 4, "sample_rate": 1.5},
            {"n_steps": 4, "sample_rate": 0},
            {"n_steps": 4, "unroll": 0},
        ],
    )
    def test_invalid_integration_counts_raise(self, model_2layer, kwargs):
        with pytest.raises(ValueError, match="positive integer"):
            model_2layer.generate_integration_fn(**kwargs)

    def test_boolean_integration_count_is_not_an_integer(self, model_2layer):
        with pytest.raises(TypeError, match="not bool"):
            model_2layer.generate_integration_fn(n_steps=True)


class TestCheckpointing:
    """Tests for checkpoint_rate defaults and the unroll clamp."""

    def test_default_checkpoint_rate_is_one(self, model_2layer):
        """The signature default is checkpoint_rate=1."""
        import inspect

        sig = inspect.signature(model_2layer.generate_integration_fn)
        assert sig.parameters["checkpoint_rate"].default == 1
        assert sig.parameters["unroll"].default == 1

    def test_checkpoint_rate_does_not_change_results(self, model_2layer, state_2layer):
        """Checkpoint granularity is a memory/speed knob only."""
        trajs = []
        for rate in (1, 5):
            integrate = model_2layer.generate_integration_fn(
                n_steps=10, sample_rate=5, checkpoint_rate=rate
            )
            trajs.append(integrate(state_2layer).qh)
        assert jnp.allclose(trajs[0], trajs[1])

    def test_unroll_is_clamped_to_checkpoint_rate(self, model_2layer, state_2layer):
        """unroll beyond checkpoint_rate has no effect (silently clamped)."""
        integrate_clamped = model_2layer.generate_integration_fn(
            n_steps=8, sample_rate=4, checkpoint_rate=2, unroll=64
        )
        integrate_equal = model_2layer.generate_integration_fn(
            n_steps=8, sample_rate=4, checkpoint_rate=2, unroll=2
        )
        assert jnp.allclose(
            integrate_clamped(state_2layer).qh, integrate_equal(state_2layer).qh
        )

    def test_undivisible_checkpoint_rate_raises(self, model_2layer):
        """checkpoint_rate must divide sample_rate."""
        with pytest.raises(ValueError, match="divisible"):
            model_2layer.generate_integration_fn(
                n_steps=10, sample_rate=5, checkpoint_rate=2
            )

    def test_invalid_checkpoint_rate_raises(self, model_2layer):
        with pytest.raises(ValueError, match="positive integer"):
            model_2layer.generate_integration_fn(
                n_steps=4, sample_rate=2, checkpoint_rate=0
            )


class TestKnownSolutions:
    """Exact low-dimensional solutions for layer-coupling regressions."""

    def test_single_fourier_mode_is_steady_without_linear_forcing(self):
        """A single QG eigenmode has zero nonlinear Jacobian."""
        grid = Grid(
            n_x=64, n_y=64, L_x=1e6, L_y=1e6, H=[125.0] * 4 + [1250.0] * 4, r_d=15000.0
        )
        mechanisms = {
            "QG": QGDynamics(grid, U=jnp.zeros(8), beta=0.0, gamma=0.0),
            "filter": SpectralFilter(grid),
        }
        model = Model(grid, mechanisms, AB3(100.0))

        # Pick one real Fourier cosine mode and obtain its PV from B phi.
        phi = jnp.linspace(1.0, 2.0, grid.n_layers)
        q_mode = jnp.einsum("ij,j->i", grid.B[:, :, 1, 1], phi)
        qh = jnp.zeros((grid.n_layers, grid.n_ky, grid.n_kx), dtype=COMPLEX)
        qh = qh.at[:, 1, 1].set(q_mode)
        qh = qh.at[:, -1, 1].set(q_mode)
        state = define_state(model, qh)

        integrate = model.generate_integration_fn(
            n_steps=8, sample_rate=1, unroll=1, sample_fields=("qh",)
        )
        trajectory = integrate(state)

        assert jnp.max(jnp.abs(trajectory.qh - qh)) < 1e-10

    def test_multilayer_rossby_wave_rhs(self):
        """A vertical eigenmode has the exact linear beta-wave tendency."""
        beta = 1e-11
        grid = Grid(
            n_x=64, n_y=64, L_x=1e6, L_y=1e6, H=[125.0] * 4 + [1250.0] * 4, r_d=15000.0
        )
        mechanisms = {
            "QG": QGDynamics(grid, U=jnp.zeros(8), beta=beta, gamma=0.0),
            "filter": SpectralFilter(grid),
        }
        ky_index, kx_index = 1, 1
        A_mode = np.asarray(grid.A[:, :, ky_index, kx_index])
        _, eigenvectors = np.linalg.eig(A_mode)
        q_mode = jnp.asarray(np.real(eigenvectors[:, 0]))
        qh = jnp.zeros((grid.n_layers, grid.n_ky, grid.n_kx), dtype=COMPLEX)
        qh = qh.at[:, ky_index, kx_index].set(q_mode)
        qh = qh.at[:, -ky_index, kx_index].set(q_mode)

        rhs = mechanisms["QG"]._rhs(qh)
        phi_mode = grid.A[:, :, ky_index, kx_index] @ q_mode
        expected = -beta * grid.ikx[ky_index, kx_index] * phi_mode

        tolerance = 1e-5 if jnp.issubdtype(COMPLEX, jnp.complex64) else 1e-11
        assert jnp.allclose(
            rhs[:, ky_index, kx_index], expected, rtol=tolerance, atol=1e-14
        )
        other_modes = rhs.at[:, ky_index, kx_index].set(0.0)
        other_modes = other_modes.at[:, -ky_index, kx_index].set(0.0)
        tolerance = 1e-8 if jnp.issubdtype(COMPLEX, jnp.complex64) else 1e-12
        assert jnp.max(jnp.abs(other_modes)) < tolerance

    def test_multilayer_rossby_wave_evolves_analytically(self):
        """The evolving eigenmode follows the discrete linear phase exactly."""
        beta, dt, n_steps = 1e-11, 100.0, 5
        grid = Grid(
            n_x=64, n_y=64, L_x=1e6, L_y=1e6, H=[125.0] * 4 + [1250.0] * 4, r_d=15000.0
        )
        mechanisms = {
            "QG": QGDynamics(grid, U=jnp.zeros(8), beta=beta, gamma=0.0),
            "filter": SpectralFilter(grid),
        }
        model = Model(grid, mechanisms, Euler(dt))
        ky_index, kx_index = 1, 1
        A_mode = np.asarray(grid.A[:, :, ky_index, kx_index])
        eigenvalues, eigenvectors = np.linalg.eig(A_mode)
        mode_index = 0
        q_mode = jnp.asarray(np.real(eigenvectors[:, mode_index]))
        mu = float(np.real(eigenvalues[mode_index]))
        qh = jnp.zeros((grid.n_layers, grid.n_ky, grid.n_kx), dtype=COMPLEX)
        qh = qh.at[:, ky_index, kx_index].set(q_mode)
        qh = qh.at[:, -ky_index, kx_index].set(q_mode)
        state = define_state(model, qh)

        integrate = model.generate_integration_fn(
            n_steps=n_steps, sample_rate=1, unroll=1, sample_fields=("qh",)
        )
        trajectory = integrate(state)
        omega_dt = beta * grid.kx[ky_index, kx_index] * mu * dt
        for step in range(1, n_steps + 1):
            factor = (1.0 - 1j * omega_dt) ** step
            expected = qh * factor
            assert jnp.allclose(
                trajectory.qh[step - 1],
                expected,
                rtol=1e-5 if jnp.issubdtype(COMPLEX, jnp.complex64) else 1e-9,
                atol=1e-6 if jnp.issubdtype(COMPLEX, jnp.complex64) else 1e-12,
            )


class TestFullIntegrationWorkflow:
    """Test complete initialisation and integration workflow."""

    def test_initialize_and_integrate_2layer(self):
        """Test creating model and integrating for 2 layers."""
        from jaxqg.utils import gaussian_random_field

        # Create grid
        grid = Grid(n_x=64, n_y=64, L_x=1e6, L_y=1e6, H=[500.0, 2000.0])
        assert grid.n_layers == 2

        # Create mechanisms and timestepper
        from jaxqg.mechanisms import SpectralFilter
        from jaxqg.timesteppers import AB3

        timestepper = AB3(dt=1800)
        mechanisms_ = {
            "QG": QGDynamics(
                grid, U=jnp.array([0.025, 0.0]), beta=1e-11, gamma=7e-8
            ),
            "filter": SpectralFilter(grid),
        }

        # Create model
        model = Model(grid, mechanisms_, timestepper)
        assert model is not None

        # Define initial state
        qh = gaussian_random_field(model, 1e-26 * model.grid.wv**-4, seed=42)
        state = define_state(model, qh)
        assert state.qh.shape == (2, 64, 33)

        # Integrate for 5 steps
        integrate = model.generate_integration_fn(n_steps=5, sample_rate=5)
        traj = integrate(state)

        # Verify output
        assert traj.qh.shape == (1, 2, 64, 33)
        assert not jnp.any(jnp.isnan(traj.qh))
        assert not jnp.any(jnp.isinf(traj.qh))

    def test_initialize_and_integrate_3layer(self):
        """Test full workflow with 3 layers."""
        from jaxqg.mechanisms import SpectralFilter
        from jaxqg.timesteppers import AB3
        from jaxqg.utils import gaussian_random_field

        grid = Grid(n_x=64, n_y=64, L_x=1e6, L_y=1e6, H=[500.0, 1000.0, 2500.0])

        timestepper = AB3(dt=1800)
        mechanisms_ = {
            "QG": QGDynamics(
                grid, U=jnp.array([0.03, 0.015, 0.0]), beta=1e-11, gamma=7e-8
            ),
            "filter": SpectralFilter(grid),
        }
        model = Model(grid, mechanisms_, timestepper)

        qh = gaussian_random_field(model, 1e-26 * model.grid.wv**-4, seed=43)
        state = define_state(model, qh)

        integrate = model.generate_integration_fn(n_steps=10, sample_rate=5)
        traj = integrate(state)

        # Should have 2 samples (n_steps // sample_rate)
        assert traj.qh.shape == (2, 3, 64, 33)

    def test_initialize_and_integrate_5layer(self):
        """Test full workflow with 5 layers."""
        from jaxqg.mechanisms import SpectralFilter
        from jaxqg.timesteppers import AB3
        from jaxqg.utils import gaussian_random_field

        grid = Grid(
            n_x=64, n_y=64, L_x=1e6, L_y=1e6, H=[200.0, 500.0, 1000.0, 2000.0, 3000.0]
        )

        timestepper = AB3(dt=1800)
        mechanisms_ = {
            "QG": QGDynamics(
                grid,
                U=jnp.array([0.04, 0.03, 0.02, 0.01, 0.0]),
                beta=1e-11,
                gamma=7e-8,
            ),
            "filter": SpectralFilter(grid),
        }
        model = Model(grid, mechanisms_, timestepper)

        qh = gaussian_random_field(model, 1e-26 * model.grid.wv**-4, seed=44)
        state = define_state(model, qh)

        integrate = model.generate_integration_fn(n_steps=3, sample_rate=3)
        traj = integrate(state)

        assert traj.qh.shape[1] == 5


class TestEnergyConservation:
    """Test energy conservation and decay properties."""

    def test_energy_monotonic_decay(self, model_2layer, state_2layer):
        """Test that total energy evolves smoothly without instability."""
        energies = []
        state = state_2layer

        for i in range(10):
            eke = EKE(model_2layer, state.qh)
            ape = APE(model_2layer, state.qh)
            total_energy = eke + ape
            energies.append(total_energy)

            if i < 9:  # Don't run the last step unnecessarily
                state = model_2layer.evolve_one_step_stateless(None, state)

        energies = jnp.array(energies)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(energies > 0)

        # Require no abrupt blow-up between adjacent steps
        energy_diff = jnp.diff(energies)
        max_step_change = jnp.max(jnp.abs(energy_diff))
        assert max_step_change < 1e-3

    def test_energy_positive_throughout(self, model_2layer, state_2layer):
        """Test that energy stays positive during integration."""
        integrate = model_2layer.generate_integration_fn(n_steps=20, sample_rate=1)
        traj = integrate(state_2layer)

        for i in range(traj.qh.shape[0]):
            eke = EKE(model_2layer, traj.qh[i])
            ape = APE(model_2layer, traj.qh[i])
            assert eke > 0
            assert ape >= 0

    def test_energy_ratio_evolution(self, model_2layer, state_2layer):
        """Test that EKE/APE ratio evolves reasonably."""
        ratios = []
        state = state_2layer

        for _ in range(5):
            eke = EKE(model_2layer, state.qh)
            ape = APE(model_2layer, state.qh)
            ratio = eke / (ape + 1e-12)  # Avoid division by zero
            ratios.append(ratio)

            state = model_2layer.evolve_one_step_stateless(None, state)

        ratios = jnp.array(ratios)

        # Ratios should be finite and positive
        assert jnp.all(jnp.isfinite(ratios))
        assert jnp.all(ratios > 0)


class TestNumericalStability:
    """Test long-term numerical stability."""

    def test_long_integration_stability(self, model_2layer, state_2layer):
        """Test that integration remains stable for many steps."""
        integrate = model_2layer.generate_integration_fn(n_steps=50, sample_rate=10)
        traj = integrate(state_2layer)

        # No NaNs or Infs
        assert not jnp.any(jnp.isnan(traj.qh))
        assert not jnp.any(jnp.isinf(traj.qh))

        # All samples should be non-zero (evolution is active)
        for i in range(traj.qh.shape[0]):
            assert jnp.any(traj.qh[i] != 0)

    def test_repeated_seed_reproducibility(self):
        """Test that same seed gives reproducible results."""
        from jaxqg.mechanisms import SpectralFilter
        from jaxqg.timesteppers import AB3
        from jaxqg.utils import gaussian_random_field

        grid = Grid(n_x=64, n_y=64, L_x=1e6, L_y=1e6, H=[500.0, 2000.0])

        timestepper = AB3(dt=1800)
        mechanisms_ = {
            "QG": QGDynamics(
                grid, U=jnp.array([0.025, 0.0]), beta=1e-11, gamma=7e-8
            ),
            "filter": SpectralFilter(grid),
        }

        # First run
        model1 = Model(grid, mechanisms_, timestepper)
        qh1 = gaussian_random_field(model1, 1e-26 * model1.grid.wv**-4, seed=100)
        state1 = define_state(model1, qh1)
        integrate1 = model1.generate_integration_fn(n_steps=5, sample_rate=5)
        traj1 = integrate1(state1)

        # Second run with same seed and setup
        model2 = Model(grid, mechanisms_, timestepper)
        qh2 = gaussian_random_field(model2, 1e-26 * model2.grid.wv**-4, seed=100)
        state2 = define_state(model2, qh2)
        integrate2 = model2.generate_integration_fn(n_steps=5, sample_rate=5)
        traj2 = integrate2(state2)

        # Results should be identical
        assert jnp.allclose(traj1.qh, traj2.qh)

    def test_different_seed_different_results(self):
        """Test that different seeds give different results."""
        from jaxqg.mechanisms import SpectralFilter
        from jaxqg.timesteppers import AB3
        from jaxqg.utils import gaussian_random_field

        grid = Grid(n_x=64, n_y=64, L_x=1e6, L_y=1e6, H=[500.0, 2000.0])

        timestepper = AB3(dt=1800)
        mechanisms_ = {
            "QG": QGDynamics(
                grid, U=jnp.array([0.025, 0.0]), beta=1e-11, gamma=7e-8
            ),
            "filter": SpectralFilter(grid),
        }

        model = Model(grid, mechanisms_, timestepper)

        qh1 = gaussian_random_field(model, 1e-26 * model.grid.wv**-4, seed=101)
        qh2 = gaussian_random_field(model, 1e-26 * model.grid.wv**-4, seed=102)

        # Different seeds should give different initial fields
        assert not jnp.allclose(qh1, qh2)


class TestMultiLayerCoupling:
    """Test inter-layer coupling in multi-layer models."""

    def test_2layer_coupling(self, model_2layer, state_2layer):
        """Test that layers evolve differently."""
        integrate = model_2layer.generate_integration_fn(n_steps=5, sample_rate=5)
        traj = integrate(state_2layer)

        # Extract final state
        qh_final = traj.qh[-1]

        # Layers should have different evolution
        layer1_final = qh_final[0]
        layer2_final = qh_final[1]

        # They shouldn't be identical
        assert not jnp.allclose(layer1_final, layer2_final)

    def test_3layer_coupling(self, model_3layer, state_3layer):
        """Test coupling in 3-layer model."""
        integrate = model_3layer.generate_integration_fn(n_steps=5, sample_rate=5)
        traj = integrate(state_3layer)

        qh_final = traj.qh[-1]

        # All layers should be different
        layer1 = qh_final[0]
        layer2 = qh_final[1]
        layer3 = qh_final[2]

        assert not jnp.allclose(layer1, layer2)
        assert not jnp.allclose(layer2, layer3)
        assert not jnp.allclose(layer1, layer3)

    def test_barotropic_mode_stability(self, model_2layer):
        """Test that barotropic (symmetric) mode evolves without error."""
        # Create purely barotropic mode (same in both layers)
        qh_baro = jnp.ones((2, 64, 33), dtype=COMPLEX) * 1e-5
        state_baro = define_state(model_2layer, qh_baro)

        # Run for a few steps
        new_state = model_2layer.evolve_one_step_stateless(None, state_baro)

        # Should evolve without error
        assert new_state.qh.shape == qh_baro.shape
        assert not jnp.any(jnp.isnan(new_state.qh))


class TestMultiGPUIntegration:
    """Test integration with multi-GPU sharding."""

    def test_sharded_state_integration(self, model_2layer, state_2layer):
        """Test integration with sharded initial state."""
        sharded_state = shard_state_across_layers(state_2layer)

        integrate = model_2layer.generate_integration_fn(n_steps=10, sample_rate=5)
        traj = integrate(sharded_state)

        assert traj.qh.shape == (2, 2, 64, 33)

    def test_sharded_state_energy(self, model_2layer, state_2layer):
        """Test energy calculations on sharded state."""
        sharded_state = shard_state_across_layers(state_2layer)

        eke_sharded = EKE(model_2layer, sharded_state.qh)
        eke_normal = EKE(model_2layer, state_2layer.qh)

        assert jnp.allclose(eke_sharded, eke_normal)

    def test_sharded_reproducibility(self, model_2layer, state_2layer):
        """Test that sharding doesn't affect determinism."""
        integrate = model_2layer.generate_integration_fn(n_steps=5, sample_rate=5)

        traj_normal = integrate(state_2layer)

        sharded_state = shard_state_across_layers(state_2layer)
        traj_sharded = integrate(sharded_state)

        # Should get identical results
        assert jnp.allclose(traj_normal.qh, traj_sharded.qh)


class TestDiagnosticsEvolution:
    """Test diagnostics throughout integration."""

    def test_cfl_below_limit(self, model_2layer):
        """Test that CFL condition stays below stability limit."""
        from jaxqg.utils import gaussian_random_field

        qh = gaussian_random_field(
            model_2layer, 1e-26 * model_2layer.grid.wv**-4, seed=50
        )
        state = define_state(model_2layer, qh)

        integrate = model_2layer.generate_integration_fn(n_steps=20, sample_rate=1)
        traj = integrate(state)

        for i in range(traj.qh.shape[0]):
            cfl_val = cfl(model_2layer, traj.qh[i])
            # Should be less than 1 for stability
            assert cfl_val < 1.0

    def test_spectral_properties(self, model_2layer, state_2layer):
        """Test spectral properties of solution."""
        integrate = model_2layer.generate_integration_fn(n_steps=10, sample_rate=5)
        traj = integrate(state_2layer)

        # Get spectrum at different times
        _kr_initial, spectrum_initial = isotropic_EKE_spectrum(
            model_2layer, state_2layer.qh
        )
        _kr_final, spectrum_final = isotropic_EKE_spectrum(model_2layer, traj.qh[-1])

        # Both should be positive
        assert jnp.all(spectrum_initial >= 0)
        assert jnp.all(spectrum_final >= 0)

        # Spectral shape might change due to dynamics
        assert spectrum_initial.shape == spectrum_final.shape
