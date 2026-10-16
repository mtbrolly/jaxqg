"""
Tests for jaxqg.diagnostics module.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxqg import APE, EKE, Grid, Model, cfl, isotropic_EKE_spectrum
from jaxqg.backend import COMPLEX
from jaxqg.diagnostics import EKE_spectrum


class TestKineticEnergy:
    """Test kinetic energy calculations."""

    def test_eke_positive(self, model_2layer, state_2layer):
        """Test that EKE is positive for non-zero states."""
        eke = EKE(model_2layer, state_2layer.qh)
        assert eke > 0

    def test_eke_zero_for_zero_qh(self, model_2layer):
        """Test that EKE is zero for zero PV."""
        qh_zero = jnp.zeros((2, 64, 33), dtype=COMPLEX)
        eke = EKE(model_2layer, qh_zero)
        assert jnp.allclose(eke, 0)

    def test_eke_scales_linearly(self, model_2layer, state_2layer):
        """Test that EKE scales with field magnitude."""
        eke1 = EKE(model_2layer, state_2layer.qh)
        eke2 = EKE(model_2layer, 2 * state_2layer.qh)

        # Kinetic energy scales with square of field
        assert jnp.allclose(eke2, 4 * eke1)

    def test_eke_3layer(self, model_3layer, state_3layer):
        """Test EKE for 3-layer model."""
        eke = EKE(model_3layer, state_3layer.qh)
        assert eke > 0
        assert jnp.isfinite(eke)


class TestPotentialEnergy:
    """Test potential energy calculations."""

    def test_ape_positive(self, model_2layer, state_2layer):
        """Test that APE is positive for multi-layer states."""
        ape = APE(model_2layer, state_2layer.qh)
        assert ape >= 0

    def test_ape_zero_for_barotropic_mode(self, model_2layer):
        """Test that APE is small for purely barotropic (symmetric) mode."""
        # Create symmetric PV: same in both layers
        qh_symmetric = jnp.ones((2, 64, 33), dtype=COMPLEX)
        ape = APE(model_2layer, qh_symmetric)
        tolerance = 2e-8 if jnp.issubdtype(COMPLEX, jnp.complex64) else 1e-10
        assert ape < tolerance

    def test_ape_3layer(self, model_3layer, state_3layer):
        """Test APE for 3-layer model."""
        ape = APE(model_3layer, state_3layer.qh)
        assert ape >= 0
        assert jnp.isfinite(ape)

    def test_ape_completes_total_energy_2layer(
        self, model_2layer, state_2layer, single_precision
    ):
        self._assert_energy_decomposition(
            model_2layer, state_2layer.qh, single_precision
        )

    def test_ape_completes_total_energy_3layer(
        self, model_3layer, state_3layer, single_precision
    ):
        self._assert_energy_decomposition(
            model_3layer, state_3layer.qh, single_precision
        )

    @staticmethod
    def _assert_energy_decomposition(model, qh, single_precision):
        p = jnp.fft.irfft2(model.grid._ph_from_qh(qh))
        q = jnp.fft.irfft2(qh)
        H_total = model.grid.H.sum()
        total_energy = -0.5 / H_total * sum(
            model.grid.H[i] * jnp.mean(p[i] * q[i])
            for i in range(model.grid.n_layers)
        )
        tolerance = 2e-5 if single_precision else 1e-11
        assert jnp.allclose(
            EKE(model, qh) + APE(model, qh),
            total_energy,
            rtol=tolerance,
            atol=1e-20,
        )

    def test_ape_scales_linearly(self, model_2layer, state_2layer):
        """Test that APE scales with field magnitude."""
        ape1 = APE(model_2layer, state_2layer.qh)
        ape2 = APE(model_2layer, 2 * state_2layer.qh)

        # APE scales with square of field
        assert jnp.allclose(ape2, 4 * ape1)


class TestCFLCondition:
    """Test CFL (Courant-Friedrichs-Lewy) condition."""

    def test_cfl_positive(self, model_2layer, state_2layer):
        """Test that CFL condition is positive."""
        cfl_val = cfl(model_2layer, state_2layer.qh)
        assert cfl_val > 0

    def test_cfl_stable_timestep(self, model_2layer, state_2layer):
        """Test that CFL < 1 for reasonable timesteps."""
        cfl_val = cfl(model_2layer, state_2layer.qh)
        # For stable integration, CFL should be less than 1
        assert cfl_val < 1.0

    def test_cfl_scales_with_amplitude(self, model_2layer, state_2layer):
        """Test that CFL increases with field amplitude."""
        cfl1 = cfl(model_2layer, state_2layer.qh)
        cfl2 = cfl(model_2layer, 2 * state_2layer.qh)

        # CFL should be larger for stronger flows
        assert cfl2 > cfl1

    def test_cfl_zero_perturbation_uses_mean_flow(self, model_2layer):
        """Zero perturbation velocity leaves the configured mean-flow CFL."""
        qh_zero = jnp.zeros((2, 64, 33), dtype=COMPLEX)
        cfl_val = cfl(model_2layer, qh_zero)
        qg = model_2layer.mechanisms["QG"]
        expected = max(
            float(jnp.max(jnp.abs(qg.U)))
            * model_2layer.timestepper.dt
            / model_2layer.grid.dx,
            float(jnp.max(jnp.abs(qg.V)))
            * model_2layer.timestepper.dt
            / model_2layer.grid.dy,
        )
        assert float(cfl_val) == pytest.approx(expected)

    def test_cfl_includes_mean_flow_and_uses_axis_spacing(self):
        """Mean velocity and dy affect CFL on a non-square grid."""
        from jaxqg.mechanisms import QGDynamics
        from jaxqg.timesteppers import AB3

        grid = Grid(
            n_x=32,
            n_y=16,
            L_x=1e6,
            L_y=2e6,
            H=[500.0, 2000.0],
        )
        dt = 100.0
        qg = QGDynamics(grid, U=[0.1, 0.2], V=[0.5, 0.1])
        model = Model(grid, {"QG": qg}, AB3(dt))
        qh_zero = jnp.zeros(
            (grid.n_layers, grid.n_ky, grid.n_kx), dtype=COMPLEX
        )

        expected = max(
            0.2 * dt / grid.dx,
            0.5 * dt / grid.dy,
        )
        assert float(cfl(model, qh_zero)) == pytest.approx(expected)


class TestEKESpectrum:
    """Test kinetic energy spectrum calculations."""

    def test_eke_spectrum_shape(self, model_2layer, state_2layer):
        """Test that EKE spectrum has correct shape."""
        spectrum = EKE_spectrum(model_2layer, state_2layer.qh)
        assert spectrum.ndim == 3  # (n_layers, n_ky, n_kx) - full 2D spectrum
        assert spectrum.shape[0] == 2

    def test_eke_spectrum_positive(self, model_2layer, state_2layer):
        """Test that spectrum is non-negative."""
        spectrum = EKE_spectrum(model_2layer, state_2layer.qh)
        assert jnp.all(spectrum >= 0)

    def test_eke_spectrum_integrates_to_total_eke(self, model_2layer, state_2layer):
        """Test that spectrum integrates to total EKE."""
        total_eke = EKE(model_2layer, state_2layer.qh)
        spectrum = EKE_spectrum(model_2layer, state_2layer.qh)

        # Integrate spectrum (2D spectrum in each layer)
        integrated = jnp.sum(spectrum)

        # EKE integrates the depth-averaged Parseval-corrected spectrum via
        # spectral_variance, which weights interior kx columns by 2 and the
        # first/last (real) columns by 1; sum(EKE_spectrum) weights every
        # Fourier column equally. With ~half the spectrum energy on the edge
        # columns this ratio is not a clean constant but is well bounded in
        # [2.0, 4.0] on this grid (measured 2.03-3.72 across seeds), replacing
        # the old order-of-magnitude check.
        ratio = integrated / (total_eke + 1e-30)
        assert integrated > 0
        assert total_eke > 0
        assert ratio > 2.0
        assert ratio < 4.0

    def test_eke_spectrum_3layer(self, model_3layer, state_3layer):
        """Test spectrum for 3-layer model."""
        spectrum = EKE_spectrum(model_3layer, state_3layer.qh)
        assert spectrum.shape[0] == 3
        assert jnp.all(spectrum >= 0)


class TestIsotropicEKESpectrum:
    """Test isotropic (radially-averaged) kinetic energy spectrum."""

    def test_isotropic_spectrum_matches_radial_band_reference(self, model_2layer):
        """Band sums agree with an independent host-side reference."""
        white = jnp.asarray(
            np.random.default_rng(123).standard_normal((2, 64, 64)),
            dtype=model_2layer.grid.H.dtype,
        )
        qh = jnp.fft.rfft2(white)
        kr, actual = isotropic_EKE_spectrum(model_2layer, qh)

        mode_spectrum = np.asarray(EKE_spectrum(model_2layer, qh))
        wv = np.asarray(model_2layer.grid.wv)
        kr = np.asarray(kr)
        dkx = float(model_2layer.grid.dk_x)
        dky = float(model_2layer.grid.dk_y)
        # Keep shell-edge arithmetic at the backend precision used by the bins.
        half_dkr = jnp.sqrt(
            model_2layer.grid.dk_x**2 + model_2layer.grid.dk_y**2
        ) / 2
        expected = np.stack(
            [
                np.where(
                    (wv >= np.asarray(jnp.asarray(radius) - half_dkr))
                    & (wv <= np.asarray(jnp.asarray(radius) + half_dkr)),
                    mode_spectrum,
                    0.0,
                ).sum(axis=(-1, -2))
                * radius
                * np.pi
                / (dkx * dky)
                for radius in kr
            ],
            axis=-1,
        )

        # Flat-spectrum FFTs can leave tiny roundoff-level values in weak bins.
        tolerance = 64 * np.finfo(mode_spectrum.dtype).eps * np.max(np.abs(expected))
        np.testing.assert_allclose(
            np.asarray(actual), expected, rtol=5e-5, atol=tolerance
        )

    def test_isotropic_spectrum_shape(self, model_2layer, state_2layer):
        """Test that isotropic spectrum is returned with wavenumbers."""
        kr, spectrum = isotropic_EKE_spectrum(model_2layer, state_2layer.qh)
        assert spectrum.ndim == 2  # (n_layers, n_wavenumbers)
        assert kr.ndim == 1

    def test_isotropic_spectrum_positive(self, model_2layer, state_2layer):
        """Test that isotropic spectrum is non-negative."""
        _kr, spectrum = isotropic_EKE_spectrum(model_2layer, state_2layer.qh)
        assert jnp.all(spectrum >= 0)

    def test_isotropic_spectrum_decreasing_trend(self, model_2layer, state_2layer):
        """Test that isotropic spectrum has some energy content."""
        _kr, spectrum = isotropic_EKE_spectrum(model_2layer, state_2layer.qh)
        # Should have non-zero energy at some wavenumbers
        assert jnp.any(spectrum > 0)

    def test_isotropic_spectrum_3layer(self, model_3layer, state_3layer):
        """Test isotropic spectrum for 3-layer model."""
        _kr, spectrum = isotropic_EKE_spectrum(model_3layer, state_3layer.qh)
        assert spectrum.ndim == 2
        assert spectrum.shape[0] == 3  # 3 layers
        assert jnp.all(spectrum >= 0)

    def test_isotropic_spectrum_white_noise(self, model_2layer):
        """Test isotropic spectrum on white noise field."""
        # White noise should have relatively flat spectrum
        qh_white = jnp.ones((2, 64, 33), dtype=COMPLEX)
        _kr, spectrum = isotropic_EKE_spectrum(model_2layer, qh_white)

        assert spectrum.ndim == 2
        assert jnp.all(jnp.isfinite(spectrum))


class TestDiagnosticsConsistency:
    """Test consistency of diagnostic quantities."""

    def test_total_energy(self, model_2layer, state_2layer):
        """Test that total energy is conserved notion."""
        eke = EKE(model_2layer, state_2layer.qh)
        ape = APE(model_2layer, state_2layer.qh)

        # Total energy should be positive
        total_energy = eke + ape
        assert total_energy > 0

    def test_energy_ratio_reasonable(self, model_2layer, state_2layer):
        """Test that EKE/APE ratio is reasonable."""
        eke = EKE(model_2layer, state_2layer.qh)
        ape = APE(model_2layer, state_2layer.qh)

        # Both should be positive
        assert eke > 0
        assert ape > 0

        # Ratio should be finite and positive
        ratio = eke / ape
        assert jnp.isfinite(ratio)
        assert ratio > 0

    def test_all_diagnostics_finite(self, model_2layer, state_2layer):
        """Test that all diagnostics return finite values."""
        eke = EKE(model_2layer, state_2layer.qh)
        ape = APE(model_2layer, state_2layer.qh)
        cfl_val = cfl(model_2layer, state_2layer.qh)
        _kr, spectrum = isotropic_EKE_spectrum(model_2layer, state_2layer.qh)

        assert jnp.isfinite(eke)
        assert jnp.isfinite(ape)
        assert jnp.isfinite(cfl_val)
        assert jnp.all(jnp.isfinite(spectrum))


class TestDiagnosticsAfterIntegration:
    """Test diagnostics on integrated states."""

    def test_diagnostics_after_step(self, model_2layer, state_2layer):
        """Test that diagnostics work after a timestep."""
        new_state = model_2layer.evolve_one_step_stateless(None, state_2layer)

        eke = EKE(model_2layer, new_state.qh)
        ape = APE(model_2layer, new_state.qh)
        cfl_val = cfl(model_2layer, new_state.qh)

        assert jnp.isfinite(eke)
        assert jnp.isfinite(ape)
        assert jnp.isfinite(cfl_val)

    def test_energy_evolution(self, model_2layer, state_2layer):
        """Test that energy evolves smoothly."""
        energies = []
        state = state_2layer

        for _ in range(5):
            eke = EKE(model_2layer, state.qh)
            energies.append(eke)
            state = model_2layer.evolve_one_step_stateless(None, state)

        energies = jnp.array(energies)

        # All should be positive and finite
        assert jnp.all(energies > 0)
        assert jnp.all(jnp.isfinite(energies))

        # Energy should evolve monotonically (decreasing with dissipation)
        assert jnp.all(jnp.diff(energies) <= 0)


def _reference_pencil_eigenvalues(H, wv_d):
    """Eigenvalues of the symmetrised stretching pencil (numpy reference).

    Builds S (stretching matrix, S = B + k²I convention) from H and the
    deformation radius, then symmetrises S_sym = D^(1/2) S D^(-1/2).
    """
    H = np.asarray(H, dtype=np.float64)
    n = len(H)
    k_d2 = wv_d**2
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                diag = 0.0
                if i > 0:
                    diag -= k_d2 * H[i - 1] / (H[i - 1] + H[i])
                if i < n - 1:
                    diag -= k_d2 * H[i + 1] / (H[i] + H[i + 1])
                S[i, i] = diag
            elif j == i + 1:
                S[i, j] = k_d2 * H[j] / (H[i] + H[j])
            elif j == i - 1:
                S[i, j] = k_d2 * H[i - 1] / (H[i - 1] + H[i])
    D_sqrt = np.sqrt(H)
    S_sym = S * D_sqrt[:, None] / D_sqrt[None, :]
    return np.linalg.eigvalsh(S_sym)


class TestVerticalModes:
    """Test the corrected vertical mode eigenproblem."""

    def test_2layer_first_baroclinic_radius(self, model_2layer, single_precision):
        """For 2 layers, the first radius equals the requested radius."""
        from jaxqg import vertical_modes

        _modes, radii = vertical_modes(model_2layer)
        target = 1.0 / model_2layer.grid.wv_d
        assert radii[0] == jnp.inf
        rel = 2e-5 if single_precision else 1e-12
        assert float(radii[1]) == pytest.approx(target, rel=rel)

    def test_large_deformation_radius_keeps_baroclinic_mode(self, single_precision):
        """A small dimensional eigenvalue is not mistaken for the barotropic zero."""
        from jaxqg import vertical_modes

        grid = Grid(
            n_x=8,
            L_x=1e6,
            L_y=1e6,
            r_d=1e8,
            H=[500.0, 2000.0],
        )
        _modes, radii = vertical_modes(SimpleNamespace(grid=grid))

        assert radii[0] == jnp.inf
        tolerance = 2e-5 if single_precision else 1e-10
        assert float(radii[1]) == pytest.approx(grid.r_d, rel=tolerance)

    def test_2layer_mode_ratio(self, model_2layer, single_precision):
        """First baroclinic mode ratio is -H_2/H_1 (ψ_1/ψ_2)."""
        from jaxqg import vertical_modes

        modes, _radii = vertical_modes(model_2layer)
        H = model_2layer.grid.H
        psi = modes[:, 1]
        expected = -float(H[1] / H[0])
        rel = 1e-5 if single_precision else 1e-10
        assert float(psi[0] / psi[1]) == pytest.approx(expected, rel=rel)

    def test_barotropic_mode_uniform(self, model_2layer, single_precision):
        """Barotropic mode has equal streamfunction across layers."""
        from jaxqg import vertical_modes

        modes, _ = vertical_modes(model_2layer)
        psi = modes[:, 0]
        atol = 2e-5 if single_precision else 1e-12
        assert jnp.allclose(psi, psi[0], rtol=atol, atol=atol)

    def test_8layer_block_vs_numpy_reference(self, single_precision):
        """8-layer block profile radii match the numpy pencil reference."""
        from jaxqg import vertical_modes
        from jaxqg.mechanisms import QGDynamics, SpectralFilter
        from jaxqg.timesteppers import AB3

        H = [125.0] * 4 + [1250.0] * 4
        grid = Grid(n_x=32, L_x=1e6, L_y=1e6, r_d=15000.0, H=H)
        model = Model(
            grid,
            {"QG": QGDynamics(grid), "filter": SpectralFilter(grid)},
            AB3(1800.0),
        )

        _modes, radii = vertical_modes(model)
        evals = _reference_pencil_eigenvalues(H, grid.wv_d)
        # Ascending -> barotropic last; reorder barotropic first.
        evals = evals[::-1]

        assert radii[0] == jnp.inf
        for m in range(1, len(H)):
            expected = 1.0 / np.sqrt(-evals[m])
            rel = 2e-6 if single_precision else 1e-10
            assert float(radii[m]) == pytest.approx(expected, rel=rel)

    def test_modes_H_orthonormal(self, single_precision):
        """Modes are orthonormal under the H-weighted inner product."""
        from jaxqg import vertical_modes
        from jaxqg.mechanisms import QGDynamics, SpectralFilter
        from jaxqg.timesteppers import AB3

        H = [125.0] * 4 + [1250.0] * 4
        grid = Grid(n_x=32, L_x=1e6, L_y=1e6, H=H)
        model = Model(
            grid,
            {"QG": QGDynamics(grid), "filter": SpectralFilter(grid)},
            AB3(1800.0),
        )
        modes, _ = vertical_modes(model)
        gram = (modes * grid.H[:, None]).T @ modes
        atol = 1e-7 if single_precision else 1e-12
        assert jnp.allclose(gram, jnp.eye(len(H)), atol=atol)

    def test_modes_are_eigenvectors_of_stretching(self, single_precision):
        """S ψ_m = λ_m ψ_m with λ_m = -1/radii_m² for baroclinic modes."""
        from jaxqg import vertical_modes
        from jaxqg.mechanisms import QGDynamics, SpectralFilter
        from jaxqg.timesteppers import AB3

        H = [125.0] * 4 + [1250.0] * 4
        grid = Grid(n_x=32, L_x=1e6, L_y=1e6, H=H)
        model = Model(
            grid,
            {"QG": QGDynamics(grid), "filter": SpectralFilter(grid)},
            AB3(1800.0),
        )
        modes, radii = vertical_modes(model)

        S = jnp.zeros((grid.n_layers, grid.n_layers))
        for i in range(grid.n_layers):
            S = S.at[i, i].set(-grid.F[i])
            if i < grid.n_layers - 1:
                S = S.at[i, i + 1].set(grid.F_interface_up[i])
                S = S.at[i + 1, i].set(grid.F_interface_down[i])

        scale = float(jnp.max(jnp.abs(S)))
        atol = 1e-8 * scale if single_precision else 1e-15 * scale
        for m in range(1, len(H)):
            lam = -1.0 / float(radii[m]) ** 2
            assert jnp.allclose(S @ modes[:, m], lam * modes[:, m], atol=atol)

    def test_modal_amplitude_roundtrip(self, model_2layer):
        """modes @ modal_amplitude(qh) reconstructs qh (H-weighted)."""
        from jaxqg import modal_amplitude, vertical_modes

        qh = (
            jax.random.normal(
                jax.random.PRNGKey(0),
                (2, model_2layer.grid.n_ky, model_2layer.grid.n_kx),
                dtype=COMPLEX,
            )
            * 1e-20
        )
        modes, _ = vertical_modes(model_2layer)
        qh_modal = modal_amplitude(model_2layer, qh, modes=modes)
        qh_recovered = jnp.einsum("im,mkl->ikl", modes, qh_modal)
        tolerance = 1e-5 if jnp.issubdtype(COMPLEX, jnp.complex64) else 1e-10
        assert jnp.allclose(qh_recovered, qh, rtol=tolerance, atol=1e-32)
