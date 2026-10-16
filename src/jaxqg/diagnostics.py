"""
A module for computing spatial statistics of a `Model` instance.
"""

import jax.numpy as jnp
import numpy as np

from .utils import spectral_variance


def EKE(model, qh):
    """Calculate depth-averaged eddy kinetic energy per unit mass in m^2/s^2.
    """
    ph = model.grid._ph_from_qh(qh)
    H_total = model.grid.H.sum()

    # Sum over all layers with thickness weighting
    eke_sum = 0.0
    for i in range(model.grid.n_layers):
        eke_sum += model.grid.H[i] * spectral_variance(model, model.grid.wv * ph[i])

    return 0.5 / H_total * eke_sum


def EQ(model, qh):
    """Calculate depth-averaged eddy potential enstrophy in 1/s^2.
    """
    H_total = model.grid.H.sum()

    # Sum over all layers with thickness weighting
    eq_sum = 0.0
    for i in range(model.grid.n_layers):
        eq_sum += model.grid.H[i] * spectral_variance(model, qh[i])

    return 0.5 / H_total * eq_sum


def APE(model, qh):
    """Calculate depth-averaged available potential energy per unit mass.

    Each interface contributes according to its coupling coefficient and the
    thickness of the adjacent layer, normalised by total model depth.
    """
    ph = model.grid._ph_from_qh(qh)
    p = jnp.fft.irfft2(ph)

    ape = 0.0
    for i in range(model.grid.n_layers - 1):
        interface_energy = (
            model.grid.H[i]
            * model.grid.F_interface_up[i]
            / model.grid.H.sum()
        )
        ape += 0.5 * interface_energy * jnp.mean((p[i] - p[i + 1]) ** 2)

    return ape


def eddy_turnover_time(model, qh):
    """Calculate eddy turnover time in days."""
    return 2.0 * jnp.pi / jnp.sqrt(EQ(model, qh)) / 86400


def cfl(model, qh):
    """Calculate Courant-Friedrichs-Lewy number.

    Maximum over all layers.
    """
    ph = model.grid._ph_from_qh(qh)
    uh = -model.grid.iky * ph
    vh = model.grid.ikx * ph
    u = jnp.fft.irfft2(uh)
    v = jnp.fft.irfft2(vh)
    qg = model.mechanisms["QG"]
    cfl_u = jnp.max(jnp.abs(u + qg.U[:, None, None]))
    cfl_v = jnp.max(jnp.abs(v + qg.V[:, None, None]))
    return jnp.maximum(
        cfl_u * model.timestepper.dt / model.grid.dx,
        cfl_v * model.timestepper.dt / model.grid.dy,
    )


def EKE_spectrum(model, qh):
    """Calculate the eddy kinetic energy spectrum for each layer.

    Returns
    -------
    spectrum : array_like, shape (n_layers, n_ky, n_kx)
        Energy spectrum for each layer
    """
    ph = model.grid._ph_from_qh(qh)
    normalisation = float(model.grid.n_x * model.grid.n_y) ** 2
    return model.grid.wv2 * jnp.abs(ph) ** 2 / normalisation


def EQ_spectrum(model, qh):
    """Calculate eddy potential enstrophy spectrum for each layer.

    Returns
    -------
    spectrum : array_like, shape (n_layers, n_ky, n_kx)
        Enstrophy spectrum for each layer
    """
    normalisation = float(model.grid.n_x * model.grid.n_y) ** 2
    return jnp.abs(qh) ** 2 / normalisation


def isotropic_spectrum(model, spectrum):
    """Calculate an isotropic spectrum from a 2D spectrum.

    Works for any number of layers.

    Parameters
    ----------
    model : Model
        Model instance
    spectrum : array_like, shape (n_layers, n_ky, n_kx)
        2D spectrum for each layer

    Returns
    -------
    kr : array_like, shape (n_radial,)
        Radial wavenumbers
    iso_spec : array_like, shape (n_layers, n_radial)
        Isotropic spectrum for each layer
    """
    kmax = model.grid.kx.max()
    dkr = jnp.sqrt(model.grid.dk_x**2 + model.grid.dk_y**2)
    kr = jnp.arange(dkr / 2.0, kmax + dkr, dkr)
    iso_spec = jnp.zeros((model.grid.n_layers, kr.size))

    for i in range(kr.size):
        iso_spec = iso_spec.at[:, i].set(
            jnp.where(
                (model.grid.wv >= kr[i] - dkr / 2) & (model.grid.wv <= kr[i] + dkr / 2),
                spectrum,
                0,
            ).sum(axis=(-1, -2))
            * kr[i]
            * jnp.pi
            / (model.grid.dk_x * model.grid.dk_y)
        )

    return kr, iso_spec


def isotropic_EKE_spectrum(model, qh):
    """Calculate isotropic energy spectrum.

    Returns
    -------
    kr : array_like, shape (n_radial,)
        Radial wavenumbers
    iso_spec : array_like, shape (n_layers, n_radial)
        Isotropic energy spectrum for each layer
    """
    return isotropic_spectrum(model, EKE_spectrum(model, qh))


def height_averaged_isotropic_EKE_spectrum(model, qh):
    """Calculate the height-averaged isotropic kinetic energy spectrum.

    Averages over all layers weighted by layer thickness.
    """
    kr, iso_spec = isotropic_EKE_spectrum(model, qh)
    H_total = model.grid.H.sum()

    # Thickness-weighted average
    h_averaged = jnp.zeros_like(kr)
    for i in range(model.grid.n_layers):
        h_averaged += model.grid.H[i] * iso_spec[i]
    h_averaged /= H_total

    return kr, h_averaged


def isotropic_EQ_spectrum(model, qh):
    """Calculate isotropic potential enstrophy spectrum.

    Returns
    -------
    kr : array_like, shape (n_radial,)
        Radial wavenumbers
    iso_spec : array_like, shape (n_layers, n_radial)
        Isotropic enstrophy spectrum for each layer
    """
    return isotropic_spectrum(model, EQ_spectrum(model, qh))


def height_averaged_isotropic_EQ_spectrum(model, qh):
    """Calculate the height-averaged isotropic eddy potential enstrophy
    spectrum.

    Averages over all layers weighted by layer thickness.
    """
    kr, iso_spec = isotropic_EQ_spectrum(model, qh)
    H_total = model.grid.H.sum()

    # Thickness-weighted average
    h_averaged = jnp.zeros_like(kr)
    for i in range(model.grid.n_layers):
        h_averaged += model.grid.H[i] * iso_spec[i]
    h_averaged /= H_total

    return kr, h_averaged


def vertical_modes(model):
    """Compute vertical modes (eigenvectors of the stretching matrix).

    For an N-layer model, the vertical modes solve the eigenproblem of the
    stretching matrix S (S = B + k²I at each wavenumber, i.e.
    qh = (∇² + S) ph), ordered by their associated deformation radii.

    The matrix S is non-symmetric for unequal layer thicknesses
    (F_interface_up != F_interface_down), so `eigh` must not be applied to
    it directly (it would silently analyse one triangle). With
    D = diag(H), the matrix L = D·S is symmetric, and the correct
    symmetric eigenproblem is S_sym = D^(-1/2) L D^(-1/2) = D^(1/2) S D^(-1/2),
    which is similar to S (same eigenvalues). The physical streamfunction
    modes are ψ = D^(-1/2) w for eigenvectors w of S_sym.

    Mode 0 (barotropic) has infinite deformation radius (eigenvalue = 0).
    Modes 1 to N-1 (baroclinic) have negative eigenvalues λ and deformation
    radii 1/sqrt(-λ).

    Parameters
    ----------
    model : Model
        Model instance

    Returns
    -------
    modes : ndarray, shape (n_layers, n_layers)
        Physical modes as columns. modes[:, n] is the n-th vertical mode.
        Modes are orthonormal under the H-weighted inner product:
        sum_i H_i modes[i, m] modes[i, n] = δ_mn, and each mode's sign is
        fixed so that its largest-magnitude entry is positive.
    radii : ndarray, shape (n_layers,)
        Deformation radii for each mode in meters.
        radii[0] = ∞ (barotropic), radii[1:] are descending baroclinic radii.
    """
    n_layers = model.grid.n_layers

    # Construct the stretching matrix S from interface coupling on the host
    # (numpy). The matrix is only n_layers x n_layers and ``jnp.linalg.eigh``
    # is not reliably supported on the Metal backend, so a host-side solve of
    # the symmetrised pencil is backend-independent and exact (float64).
    H = np.asarray(model.grid.H, dtype=np.float64)
    F_dn = np.asarray(model.grid.F_interface_down, dtype=np.float64)
    F_up = np.asarray(model.grid.F_interface_up, dtype=np.float64)
    S = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        diag_val = 0.0
        if i > 0:
            diag_val -= F_dn[i - 1]
        if i < n_layers - 1:
            diag_val -= F_up[i]
        S[i, i] = diag_val
        if i > 0:
            S[i, i - 1] = F_dn[i - 1]
        if i < n_layers - 1:
            S[i, i + 1] = F_up[i]

    # Symmetrised pencil: S_sym = D^(1/2) S D^(-1/2), similar to S
    D_sqrt = np.sqrt(H)
    S_sym = S * D_sqrt[:, None] / D_sqrt[None, :]
    evals, evecs = np.linalg.eigh(S_sym)

    # eigh returns ascending eigenvalues (most-negative baroclinic first,
    # barotropic zero last). Reorder barotropic first, then descending
    # baroclinic radii (eigenvalues increasing in magnitude).
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    # Physical streamfunction modes ψ = D^(-1/2) w (H-orthonormal by
    # construction since the w are orthonormal).
    modes = evecs / D_sqrt[:, None]

    # Canonical sign: largest-magnitude entry of each mode is positive.
    idx = np.argmax(np.abs(modes), axis=0)
    signs = np.sign(modes[idx, np.arange(n_layers)])
    modes = modes * signs[None, :]

    # Clamp only zero eigenvalues at the eigensolver's relative precision.
    eigenvalue_scale = np.max(np.abs(evals))
    zero_tolerance = (
        10.0 * np.finfo(evals.dtype).eps * n_layers * eigenvalue_scale
    )
    evals = np.where(np.abs(evals) <= zero_tolerance, 0.0, evals)

    # Compute deformation radii from eigenvalues
    # For eigenvalue λ < 0, the deformation radius is R_d = 1/sqrt(-λ)
    radii = np.full(n_layers, np.inf)
    for i in range(n_layers):
        if evals[i] < 0.0:
            radii[i] = 1.0 / np.sqrt(-evals[i])

    return jnp.asarray(modes), jnp.asarray(radii)


def modal_amplitude(model, qh, modes=None):
    """Project PV onto vertical modes to get modal amplitudes.

    Parameters
    ----------
    model : Model
        Model instance
    qh : array_like, shape (n_layers, n_ky, n_kx)
        Potential vorticity in Fourier space
    modes : ndarray, optional
        Eigenvectors from vertical_modes(). If None, computed here.

    Returns
    -------
    qh_modal : array_like, shape (n_layers, n_ky, n_kx)
        Mode amplitudes in Fourier space. qh_modal[m] is the amplitude
        of mode m at each wavenumber.

    Notes
    -----
    The transformation uses the H-weighted inner product under which the
    modes from vertical_modes() are orthonormal:
    qh_modal[m] = sum_i H_i modes[i, m] qh[i].
    The inverse transformation is qh = modes @ qh_modal.
    """
    if modes is None:
        modes, _ = vertical_modes(model)

    # H-weighted projection: qh_modal[m, ky, kx] = sum_i H_i modes[i, m] qh[i, ky, kx]
    weighted_modes = modes * model.grid.H[:, None]
    qh_modal = jnp.einsum("im,ikl->mkl", weighted_modes, qh)

    return qh_modal
