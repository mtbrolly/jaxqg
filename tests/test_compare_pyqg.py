"""
Cross-implementation validation against pyqg (the reference QG model).

These are the strongest regression tests for the N > 2 (arbitrary-layer)
formulation: for identical layer thicknesses, base-state flow, beta, bottom
drag, and vertical coupling, jaxqg's pseudo-spectral tendency must agree with
pyqg's to floating-point precision.

pyqg is an optional reference dependency. Where it is not installed these
tests are skipped (``pytest.importorskip``), so the suite still runs in
minimal environments. On CUDA/CPU with 64-bit precision enabled (the default
outside Apple Metal) the comparison is exact; tolerances are set for float64.

Equivalence of the two formulations
-----------------------------------
pyqg's layered model evolves (layered_model.py docstring)::

    dq/dt = -J(psi, q) - (U + u) dq/dx - v dq/dy
            - Qy psi_x + Qx psi_y - rek delta_{iN} lap2 psi

with the mean PV gradients ``Qy = beta*1 - S @ U`` and ``Qx = S @ V``, where
``S`` is the stretching matrix and ``q = (lap2 + S) psi``.

jaxqg's ``QGDynamics._rhs`` computes::

    advection   = -ikx*fft((U + u)q) - iky*fft(v q)     # -J + U advection
    beta_term   = -beta_eff * ikx * ph,  beta_eff = beta - S@U
    meridional  = -Qx * iky * ph,        Qx = S@V
    bottom drag = gamma * wv2 * ph (bottom layer only)

These are the same equation. Two implementation notes that shape the test
setup:

1. **Vertical coupling.** jaxqg builds its stretching matrix from a
   deformation radius ``r_d`` and layer thicknesses ``H``; pyqg builds ``S``
   from layer densities ``rho`` and reduced gravities. To make the comparison
   exact and independent of pyqg's ``rho``/``gpi`` parameterisation, we
   overwrite pyqg's ``S`` with the matrix implied by jaxqg's coupling and
   rebuild the derived fields (``Qy``, inversion matrix ``a``) from it.

2. **Meridional background flow (V).** pyqg's *documented* equation includes
   the ``-Qx psi_y`` term, but its compiled kernel only applies the zonal PV
   gradient (``_ikQy``); the ``v q`` advection uses the eddy velocity ``v``
   alone, without ``V``. jaxqg implements the full meridional flow. We
   therefore compare with ``V = 0`` (the case pyqg actually supports), leaving
   jaxqg's meridional term identically zero as well.
"""

import numpy as np
import pytest

pyqg = pytest.importorskip("pyqg")

import jax.numpy as jnp

from jaxqg import Grid, Model, define_state
from jaxqg.mechanisms import QGDynamics, SpectralFilter
from jaxqg.timesteppers import AB3


def _stretching_matrix_from_jaxqg(grid):
    """Recover the pyqg-format stretching matrix ``S`` from a jaxqg Grid.

    jaxqg's inversion relation is ``qh = B ph`` with ``B = -wv2*I - F``,
    while pyqg's is ``qh = (S - wv2*I) ph``. Hence ``S = -F = B + wv2*I``.
    Reading ``grid.B`` at any nonzero wavenumber and adding ``wv2*I`` gives the
    wavenumber-independent ``S`` shared by both models.
    """
    wv2 = float(np.asarray(grid.wv2)[1, 1])
    B = np.asarray(grid.B[:, :, 1, 1])
    # B is complex-typed but numerically real; return a float64 S (pyqg's
    # compiled buffers reject complex dtypes).
    return np.real(B) + wv2 * np.eye(grid.n_layers)


def _build_equivalent_pyqg(grid, U, V, beta, rek, dt=1800.0):
    """Build a pyqg.LayeredModel whose physics matches `grid` + dynamics."""
    nz = grid.n_layers
    # Parameters `g`/`f`/`rho` are required for pyqg to build *some* vertical
    # coupling, but we immediately override the resulting stretching matrix S
    # below, so their values are irrelevant to the comparison. They must still
    # be valid (increasing rho for positive reduced gravities, truthy f so
    # pyqg computes f2) or construction errors before we can override S.
    rho = np.linspace(1025.0, 1026.0, nz)
    m = pyqg.LayeredModel(
        g=9.81,
        beta=beta,
        nz=nz,
        rd=15000.0,
        H=np.asarray(grid.H, dtype=np.float64),
        U=np.asarray(U, dtype=np.float64),
        V=np.asarray(V, dtype=np.float64),
        rho=rho,
        f=1.0e-4,
        nx=grid.n_x,
        ny=grid.n_y,
        L=grid.L_x,
        W=grid.L_y,
        rek=rek,
        dt=dt,
        log_level=0,
    )

    # Match jaxqg's vertical coupling exactly.
    m.S = _stretching_matrix_from_jaxqg(grid)

    # Rebuild the mean PV gradient (the Qy setter also updates the compiled
    # ``_ikQy`` buffer used by ``_do_advection``).
    m.Qy = beta - m.S @ m.Ubg

    # Rebuild the inversion matrix a (psi = a qh) from the new S.
    m._initialize_inversion_matrix()

    return m


def _jaxqg_model(nx, ny, H, U, V, beta, gamma):
    grid = Grid(n_x=nx, n_y=ny, L_x=1e6, L_y=1e6, H=H, r_d=15000.0)
    qg = QGDynamics(
        grid, U=jnp.asarray(U), V=jnp.asarray(V), beta=beta, gamma=gamma
    )
    model = Model(grid, {"QG": qg, "filter": SpectralFilter(grid)}, AB3(1800.0))
    return grid, qg, model


def _random_initial_qh(nz, ny, nkx, seed, amplitude=1e-7):
    """Smooth, valid-rfft, zero-mean PV with a fixed seed.

    ``amplitude`` is the peak real-space |PV| (jaxqg/pyqg PV anomalies are
    ~1e-7); O(1) random fields are unrealistically energetic and blow up under
    the baroclinic instability of the block profile.

    The field is drawn in real space and FFT'd (as jaxqg's own
    ``gaussian_random_field`` does). This guarantees a *valid* rfft field
    (``rfft2(irfft2(qh)) == qh`` to machine precision) and, via a red
    ``wv**-2`` spectrum, a smooth large-scale field, then renormalised to the
    target amplitude. Building ``qh`` directly in spectral space with a naive
    ``conj(qh[:, ::-1, ::-1])`` symmetrisation is subtly wrong for the rfft2
    half-plane layout (off-by-one reflection) and leaves an aliasing/striping
    artefact.
    """
    nx = (nkx - 1) * 2  # nkx == nx//2 + 1 for an rfft2 layout
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((nz, ny, nx))
    qh = np.fft.rfft2(white)  # (nz, ny, nkx), valid rfft field
    kv = np.fft.fftfreq(ny)[:, None] ** 2 + np.fft.rfftfreq(nx)[None, :] ** 2
    kv[0, 0] = 1.0
    qh = qh / np.sqrt(kv)  # red / large-scale amplitude (~wv^-1)
    qh[:, 0, 0] = 0.0  # remove the domain-mean mode
    peak = float(np.max(np.abs(np.fft.irfft2(qh))))
    return qh * (amplitude / peak)  # complex128, normalised to target PV


# Layer profiles for N > 2. Unequal thicknesses exercise the non-symmetric S
# (F_interface_up != F_interface_down) and stay far from the 2-layer special
# case.
PROFILES = {
    3: [300.0, 700.0, 2000.0],
    4: [200.0, 500.0, 1000.0, 2500.0],
    8: [125.0] * 4 + [1250.0] * 4,
}


@pytest.mark.parametrize("nz", [3, 4, 8])
def test_inversion_matrix_matches_pyqg(nz):
    """jaxqg's A (psi = A qh) equals pyqg's a (real-valued)."""
    H = PROFILES[nz]
    grid, _, _ = _jaxqg_model(
        32, 32, H, U=np.zeros(nz), V=np.zeros(nz), beta=0.0, gamma=0.0
    )
    m = _build_equivalent_pyqg(grid, U=np.zeros(nz), V=np.zeros(nz), beta=0.0, rek=0.0)

    A_jaxqg = np.asarray(grid.A).real  # A is complex-typed but numerically real
    a_pyqg = np.asarray(m.a)

    assert A_jaxqg.shape == a_pyqg.shape
    # Both are float64 and derive from the same S, but via different inversion
    # orderings (jaxqg tiles over ky, pyqg solves the whole batch at once), so
    # small numerical-path differences (up to ~1e-6 for some profiles) are
    # allowed. The tendency test below is the authoritative, tighter check.
    np.testing.assert_allclose(A_jaxqg, a_pyqg, rtol=1e-4, atol=1e-10)


@pytest.mark.parametrize("nz", [3, 4, 8])
def test_full_nonlinear_tendency_matches_pyqg(nz):
    """The complete dq/dt (inversion + advection + beta + shear + drag)
    matches pyqg's compiled kernel for an arbitrary initial qh."""
    nz = int(nz)
    H = PROFILES[nz]
    beta, gamma = 1e-11, 7e-8
    U = np.linspace(0.05, 0.0, nz)
    V = np.zeros(nz)  # pyqg's kernel does not apply meridional background flow

    grid, qg, _ = _jaxqg_model(32, 32, H, U=U, V=V, beta=beta, gamma=gamma)
    m = _build_equivalent_pyqg(grid, U=U, V=V, beta=beta, rek=gamma)

    qh = _random_initial_qh(nz, grid.n_ky, grid.n_kx, seed=1)

    # jaxqg tendency.
    rhs_jaxqg = np.asarray(qg._rhs(jnp.asarray(qh)))

    # pyqg tendency. The ``qh`` property setter refreshes the real-space ``q``
    # via ``ifft_qh_to_q``; then invert + advect + friction fill ``dqhdt``.
    m.qh = qh.copy()
    m._invert()
    m._do_advection()
    m._do_friction()
    rhs_pyqg = np.asarray(m.dqhdt).copy()

    # Identical equation and (unnormalised) FFT conventions => machine
    # precision agreement in float64.
    scale = np.max(np.abs(rhs_jaxqg))
    np.testing.assert_allclose(
        rhs_jaxqg, rhs_pyqg, rtol=1e-10, atol=1e-12 * max(scale, 1.0)
    )


@pytest.mark.parametrize("nz", [3, 4, 8])
def test_multistep_integration_matches_pyqg(nz):
    """A full AB3 + spectral-filter integration (many steps) matches pyqg.

    The single-step tendency check does not exercise the timestepper's
    startup sequence (Euler -> AB2 -> AB3), the tendency-history bookkeeping,
    or the accumulation of round-off across steps. Driving both models through
    the same number of identical steps (far beyond the AB3 startup) and
    comparing the final PV closes that gap. This uses jaxqg's default AB3
    timestepper and SpectralFilter, matching pyqg's default AB3 and filter.
    """
    nz = int(nz)
    H = PROFILES[nz]
    beta, gamma = 1e-11, 7e-8
    U = np.linspace(0.05, 0.0, nz)
    V = np.zeros(nz)  # pyqg's kernel ignores meridional background flow

    grid, _qg, model = _jaxqg_model(32, 32, H, U=U, V=V, beta=beta, gamma=gamma)
    m = _build_equivalent_pyqg(grid, U=U, V=V, beta=beta, rek=gamma, dt=1800.0)

    qh = _random_initial_qh(nz, grid.n_ky, grid.n_kx, seed=2)
    n_steps = 120  # well past the 3-step AB3 startup; accumulates any mismatch

    # jaxqg: explicit loop over the (stateless) evolve step so this also runs
    # on older JAX versions that lack the current ``lax.scan`` signature.
    state = define_state(model, jnp.asarray(qh))
    for _ in range(n_steps):
        state = model.evolve_one_step_stateless(None, state)
    qh_jaxqg = np.asarray(state.qh)

    # pyqg: drive the same number of complete steps. Each ``_step_forward``
    # inverts, advects, applies friction and filter, and advances via AB3 with
    # the same Euler->AB2->AB3 startup as jaxqg.
    m.qh = qh.copy()
    for _ in range(n_steps):
        m._step_forward()
    qh_pyqg = np.asarray(m.qh).copy()

    scale = np.max(np.abs(qh_jaxqg))
    np.testing.assert_allclose(
        qh_jaxqg, qh_pyqg, rtol=1e-9, atol=1e-11 * max(scale, 1.0)
    )
