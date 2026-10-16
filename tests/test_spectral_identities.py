"""
Spectral identity tests inspired by pyqg's FFT/inversion test style.
"""

import jax.numpy as jnp

from jaxqg.backend import COMPLEX, FLOAT
from jaxqg.diagnostics import spectral_variance


def test_rfft_parseval_half_plane_weighting(grid_2layer):
    """Validate Parseval consistency for real FFT half-plane conventions."""
    n_layers = grid_2layer.n_layers
    field = jnp.zeros((n_layers, grid_2layer.n_y, grid_2layer.n_x), dtype=FLOAT)
    field = field.at[0].set(
        jnp.cos(2.0 * jnp.pi * 3 * grid_2layer.x / grid_2layer.L_x)
        + 0.5 * jnp.sin(2.0 * jnp.pi * 2 * grid_2layer.y / grid_2layer.L_y)
    )

    fh = jnp.fft.rfft2(field)

    class DummyModel:
        grid = grid_2layer

    var_spec = spectral_variance(DummyModel(), fh)
    var_real = jnp.mean(field**2, axis=(-2, -1))

    rtol, atol = (1e-6, 1e-6) if FLOAT == jnp.float32 else (1e-12, 1e-12)
    assert jnp.allclose(var_spec, var_real, rtol=rtol, atol=atol)


def test_forward_backward_fft_roundtrip(grid_2layer):
    """Forward then inverse real FFT should reconstruct field."""
    field = jnp.cos(
        2.0 * jnp.pi * 4 * grid_2layer.x / grid_2layer.L_x
    ) + 0.25 * jnp.sin(2.0 * jnp.pi * 5 * grid_2layer.y / grid_2layer.L_y)
    field_layers = jnp.stack([field, 0.7 * field], axis=0)

    fh = jnp.fft.rfft2(field_layers)
    reconstructed = jnp.fft.irfft2(fh)

    rtol, atol = (1e-5, 1e-6) if FLOAT == jnp.float32 else (1e-12, 1e-12)
    assert jnp.allclose(reconstructed, field_layers, rtol=rtol, atol=atol)


def test_inversion_matrix_identity_nonzero_mode(grid_3layer):
    """Check B @ A ~= I for a representative nonzero mode."""
    ky_idx = 1
    kx_idx = 2

    A = grid_3layer.A[:, :, ky_idx, kx_idx]
    B = grid_3layer.B[:, :, ky_idx, kx_idx]
    I = B @ A

    # For N>2, A comes from a numerical complex matrix inverse
    # (grid._initialise_inversion_matrix); float32 LAPACK accuracy varies
    # by platform (x86 complex32 is markedly worse than Apple's), so the
    # float32 tolerance here is looser than the analytic-path tests.
    rtol, atol = (1e-4, 1e-5) if FLOAT == jnp.float32 else (1e-10, 1e-10)
    assert jnp.allclose(I, jnp.eye(grid_3layer.n_layers), rtol=rtol, atol=atol)


def test_qh_ph_inversion_roundtrip_nonzero_mode(grid_3layer):
    """Applying q->psi then psi->q returns original q for nonzero mode content."""
    qh = jnp.zeros(
        (grid_3layer.n_layers, grid_3layer.n_ky, grid_3layer.n_kx), dtype=COMPLEX
    )
    qh = qh.at[:, 1, 2].set(jnp.array([1.0 + 2.0j, -0.5 + 0.25j, 0.75 - 1.25j]))

    ph = grid_3layer._ph_from_qh(qh)
    qh_recovered = grid_3layer._qh_from_ph(ph)

    rtol, atol = (1e-5, 1e-6) if FLOAT == jnp.float32 else (1e-10, 1e-10)
    assert jnp.allclose(qh_recovered, qh, rtol=rtol, atol=atol)


def test_barotropic_mode_inversion_consistency(grid_2layer):
    """Barotropic q (equal layers) yields equal streamfunction across layers."""
    qh = jnp.zeros((2, grid_2layer.n_ky, grid_2layer.n_kx), dtype=COMPLEX)
    qh = qh.at[0, 2, 3].set(1.0 + 0.5j)
    qh = qh.at[1, 2, 3].set(1.0 + 0.5j)

    ph = grid_2layer._ph_from_qh(qh)

    rtol, atol = (1e-6, 1e-6) if FLOAT == jnp.float32 else (1e-12, 1e-12)
    assert jnp.allclose(ph[0], ph[1], rtol=rtol, atol=atol)
