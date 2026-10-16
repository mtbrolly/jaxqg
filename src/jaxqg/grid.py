import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from .backend import COMPLEX, FLOAT, selected_platform

jnp = jax.numpy


def _under_jax_trace():
    """True when running inside a JAX trace (jit/scan/...)."""
    try:
        from jax._src.core import trace_state_clean

        return not trace_state_clean()
    except ImportError:
        return True


def _wv_d_for_r_d(H, r_d):
    """Convert a physical first deformation radius into ``wv_d``.

    The layer-coupling matrix is proportional to ``wv_d**2``. Compute the
    first baroclinic eigenvalue at unit input radius and rescale so the
    requested physical radius is the model's first mode.

    ``H`` is the layer-thickness sequence; ``r_d`` must be positive.
    """
    H = np.asarray(H, dtype=np.float64)
    n_layers = len(H)
    S = np.zeros((n_layers, n_layers), dtype=np.float64)
    for i in range(n_layers):
        if i > 0:
            f_down = H[i - 1] / (H[i - 1] + H[i])
            S[i, i - 1] = f_down
            S[i, i] -= f_down
        if i < n_layers - 1:
            f_up = H[i + 1] / (H[i] + H[i + 1])
            S[i, i + 1] = f_up
            S[i, i] -= f_up
    d_sqrt = np.sqrt(H)
    S_sym = S * d_sqrt[:, None] / d_sqrt[None, :]
    evals = np.sort(np.linalg.eigvalsh(S_sym))
    scale = max(abs(evals[0]), abs(evals[-1]))
    baroclinic = evals[evals < -1e-10 * scale]
    if baroclinic.size == 0:
        raise ValueError("at least two layers are required")
    alpha = -baroclinic[-1]
    return float(1.0 / (r_d * np.sqrt(alpha)))


def _numpy_A_tile(w, F, F_up, F_dn, fix_zero_mode=False):
    """Host-side inverse coupling matrix for one ``ky`` tile of wavenumbers.

    Given the wavenumber-squared tile ``w`` (shape ``(t, n_kx)``) and the
    layer-coupling vectors, return ``A`` (shape ``(L, L, t, n_kx)``) with
    ``A @ qh_tile = ph_tile`` at each wavenumber, i.e. the tile-wise
    inverse of the stretching operator ``B``.

    The (ky, kx) = (0, 0) block of B is singular (barotropic zero mode);
    with ``fix_zero_mode`` (first tile only) it is replaced by a
    well-conditioned block before inversion and the corresponding block
    of A is zeroed afterwards.
    """
    L = F.shape[0]
    dtype = np.dtype(COMPLEX)

    if L == 2:
        # Analytic 2-layer inverse (pyqg formulation). The zero mode has
        # det = 0, so its entries vanish via det_inv = 0.
        det = w * (w + F[0] + F[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            det_inv = 1.0 / det
        det_inv = np.where(np.isfinite(det_inv), det_inv, 0.0)
        A = np.empty((2, 2) + w.shape, dtype=dtype)
        A[0, 0] = -w - F[1]
        A[0, 1] = -F[0]
        A[1, 0] = -F[1]
        A[1, 1] = -w - F[0]
        A *= det_inv
        if fix_zero_mode:
            A[:, :, 0, 0] = 0.0
        return A

    B = np.zeros((L, L) + w.shape, dtype=dtype)
    for i in range(L):
        B[i, i] = -w - F[i]
        if i < L - 1:
            B[i, i + 1] = F_up[i]
            B[i + 1, i] = F_dn[i]
    if fix_zero_mode:
        idx = np.arange(L)
        B[:, :, 0, 0] = 0.0
        B[idx, idx, 0, 0] = 1.0
    A = np.linalg.inv(B.transpose(2, 3, 0, 1)).transpose(2, 3, 0, 1)
    if fix_zero_mode:
        A[:, :, 0, 0] = 0.0
    return A


def _tile_rows(n_rows, row_cost):
    """Split ``n_rows`` into tiles of at most ~2**26 elements times ``row_cost``."""
    tile = max(1, int(2**26 // row_cost))
    return [(lo, min(lo + tile, n_rows)) for lo in range(0, n_rows, tile)]


@register_pytree_node_class
class Grid:
    """Spatial domain with associated physical parameters."""

    def __init__(
        self,
        n_x,
        n_y=None,
        L_x=2 * jnp.pi,
        L_y=2 * jnp.pi,
        r_d=15000.0,
        H=None,
        a_shard_devices=None,
    ):
        """Create a periodic N-layer grid and its PV inversion operator.

        Parameters
        ----------
        n_x, n_y : int
            Grid resolution (``n_y`` defaults to ``n_x``); both must be even.
        L_x, L_y : float
            Domain size in x and y.
        r_d : float
            First baroclinic deformation radius (must be positive).
        H : sequence of float, optional
            Layer thicknesses. Defaults to a 2-layer ``[500, 2000]``.
        a_shard_devices : sequence of Device, optional
            Enables the experimental layer-sharded inversion path; the
            layer count must divide evenly by the device count, and
            construction memory use is not guaranteed.
        """
        if n_y is None:
            n_y = n_x
        if n_x % 2 != 0:
            raise ValueError(f"n_x must be even, got {n_x}")
        if n_y % 2 != 0:
            raise ValueError(f"n_y must be even, got {n_y}")
        self.n_x = n_x
        self.n_y = n_y
        self.L_x = L_x
        self.L_y = L_y

        if H is None:
            H = [500.0, 2000.0]
        self.H = jnp.array(H, dtype=FLOAT)
        self.n_layers = len(self.H)

        if r_d <= 0:
            raise ValueError("r_d must be positive")
        self.r_d = float(r_d)
        # For 1 layer the coupling scale is the radius itself; for N > 1
        # the physical radius fixes the first baroclinic mode exactly.
        self.wv_d = (
            1.0 / self.r_d if self.n_layers == 1 else _wv_d_for_r_d(self.H, self.r_d)
        )

        self._a_shard_devices = (
            list(a_shard_devices) if a_shard_devices is not None else None
        )
        self._B_cache = None
        self._a_state_sharding = None

        self._construct_grids()
        self._initialise_stretching_parameters()
        self._initialise_inversion_matrix()

    def _construct_grids(self):
        """Construct spatial grids in real and Fourier space."""
        self.x, self.y = jnp.meshgrid(
            self.L_x * jnp.arange(0.5, self.n_x, dtype=FLOAT) / self.n_x,
            self.L_y * jnp.arange(0.5, self.n_y, dtype=FLOAT) / self.n_y,
        )
        self.dx = self.L_x / self.n_x
        self.dy = self.L_y / self.n_y
        self.dk_x = 2 * jnp.pi / self.L_x
        self.dk_y = 2 * jnp.pi / self.L_y
        self.n_kx = self.n_x // 2 + 1
        self.n_ky = self.n_y
        self.kx = self.dk_x * jnp.arange(0.0, self.n_kx, dtype=FLOAT)
        self.ky = self.dk_y * jnp.append(
            jnp.arange(0.0, self.n_y / 2, dtype=FLOAT),
            jnp.arange(-self.n_y / 2, 0.0, dtype=FLOAT),
        )
        self.kx, self.ky = jnp.meshgrid(self.kx, self.ky)
        self.ikx = 1j * self.kx
        self.iky = 1j * self.ky
        self.wv2 = self.kx**2 + self.ky**2
        self.wv = jnp.sqrt(self.wv2)
        self.wv2i = jnp.where(self.wv2 != 0.0, self.wv2**-1, 0)
        self.wvi = jnp.where(self.wv != 0.0, self.wv**-1, 0)
        self.wv2_nd = (self.kx / self.dk_x) ** 2 + (self.ky / self.dk_y) ** 2

    def _initialise_stretching_parameters(self):
        """Compute the layer stretching parameters from the layer thicknesses.

        The stretching felt by layer i couples it to its vertical
        neighbours through the interface fractions
        ``H[j] / (H[i] + H[j])``, scaled by ``wv_d**2``:

        - Top and bottom layers: coupled only to their single neighbour.
        - Interior layers: sum of the couplings to the layers above and
          below.

        The per-interface couplings are also stored for the tridiagonal
        inversion operator:

        - ``F_interface_up[i]``: coupling from layer i to layer i+1.
        - ``F_interface_down[i]``: coupling from layer i+1 to layer i.

        For 2 layers this reduces to the standard formulation
        ``F[0] = wv_d**2 / (1 + delta)`` and
        ``F[1] = wv_d**2 * delta / (1 + delta)`` with
        ``delta = H[0] / H[1]``.
        """
        self.F = jnp.zeros(self.n_layers, dtype=FLOAT)

        if self.n_layers == 1:
            self.F_interface_up = jnp.zeros(0, dtype=FLOAT)
            self.F_interface_down = jnp.zeros(0, dtype=FLOAT)
            return

        for i in range(self.n_layers):
            if i == 0:
                # Top layer: only the interface below
                self.F = self.F.at[i].set(
                    self.wv_d**2 * self.H[1] / (self.H[0] + self.H[1])
                )
            elif i == self.n_layers - 1:
                # Bottom layer: only the interface above
                self.F = self.F.at[i].set(
                    self.wv_d**2 * self.H[i - 1] / (self.H[i - 1] + self.H[i])
                )
            else:
                # Interior layer: interfaces above and below
                F_above = self.wv_d**2 * self.H[i - 1] / (self.H[i - 1] + self.H[i])
                F_below = self.wv_d**2 * self.H[i + 1] / (self.H[i] + self.H[i + 1])
                self.F = self.F.at[i].set(F_above + F_below)

        self.F_interface_up = jnp.zeros(self.n_layers - 1, dtype=FLOAT)
        self.F_interface_down = jnp.zeros(self.n_layers - 1, dtype=FLOAT)
        for i in range(self.n_layers - 1):
            self.F_interface_up = self.F_interface_up.at[i].set(
                self.wv_d**2 * self.H[i + 1] / (self.H[i] + self.H[i + 1])
            )
            self.F_interface_down = self.F_interface_down.at[i].set(
                self.wv_d**2 * self.H[i] / (self.H[i] + self.H[i + 1])
            )

        if self.n_layers == 2:
            # Backward-compatible 2-layer aliases
            self.H_1 = self.H[0]
            self.H_2 = self.H[1]
            self.delta = self.H_1 / self.H_2
            self.F_1 = self.F[0]
            self.F_2 = self.F[1]

    def _build_B(self):
        """Build the full B matrix satisfying qh = B ph (dense, default device).

        Shape: (n_layers, n_layers, n_ky, n_kx), COMPLEX dtype.
        """
        B = jnp.zeros(
            (self.n_layers, self.n_layers, self.n_ky, self.n_kx), dtype=COMPLEX
        )
        for i in range(self.n_layers):
            B = B.at[i, i].set(-self.wv2 - self.F[i])
            if i < self.n_layers - 1:
                B = B.at[i, i + 1].set(self.F_interface_up[i])
            if i > 0:
                B = B.at[i, i - 1].set(self.F_interface_down[i - 1])
        return B

    def _numpy_coupling_constants(self):
        """Host-side (float64/complex) copies of the coupling constants."""
        return (
            np.asarray(self.wv2, dtype=np.float64),
            np.asarray(self.F, dtype=np.float64),
            np.asarray(self.F_interface_up, dtype=np.float64),
            np.asarray(self.F_interface_down, dtype=np.float64),
        )

    def _build_dense_A_host(self):
        """Build the dense inverse on the host, tiled over ``ky``.

        Used by backends without a complex solve (e.g. Metal). Tiling
        limits temporary workspace; the full dense result is transferred
        to the device at the end.
        """
        wv2, F, F_up, F_dn = self._numpy_coupling_constants()
        L = self.n_layers
        itemsize = np.dtype(COMPLEX).itemsize

        A_host = np.empty((L, L, self.n_ky, self.n_kx), dtype=np.dtype(COMPLEX))
        for lo, hi in _tile_rows(
            self.n_ky, L * L * self.n_kx * itemsize
        ):
            A_host[:, :, lo:hi, :] = _numpy_A_tile(
                wv2[lo:hi], F, F_up, F_dn, fix_zero_mode=(lo == 0)
            )
        return jnp.asarray(A_host)

    def _build_sharded_A(self, devices):
        """Build the layer-sharded inversion array in ``ky`` tiles.

        Each device ends up owning a contiguous block of layer rows of A,
        so the einsum in `_ph_from_qh` can keep A's rows local.
        """
        mesh = Mesh(np.array(devices), axis_names=("layers",))
        a_sharding = NamedSharding(mesh, P("layers", None, None, None))
        # Sharding for state-shaped (n_layers, n_ky, n_kx) arrays of the
        # einsum output in _ph_from_qh (pins GSPMD to keep A's rows local).
        self._a_state_sharding = NamedSharding(mesh, P("layers", None, None))

        wv2, F, F_up, F_dn = self._numpy_coupling_constants()
        L = self.n_layers

        def callback(index):
            lo, hi = index[0].start, index[0].stop
            block = np.empty((hi - lo, L, self.n_ky, self.n_kx), np.dtype(COMPLEX))
            for ky_lo, ky_hi in _tile_rows(self.n_ky, L * L * self.n_kx):
                A_tile = _numpy_A_tile(
                    wv2[ky_lo:ky_hi], F, F_up, F_dn, fix_zero_mode=(ky_lo == 0)
                )
                block[:, :, ky_lo:ky_hi, :] = A_tile[lo:hi]
            return block

        shape = (L, L, self.n_ky, self.n_kx)
        return jax.make_array_from_callback(shape, a_sharding, callback)

    def _initialise_inversion_matrix(self):
        """Compute the matrix A satisfying ph = A qh.

        A is an (n_layers, n_layers) matrix at each wavenumber that
        inverts the tridiagonal stretching operator
        ``B[i, i] = -k**2 - F[i]``, ``B[i, i±1]`` the interface couplings.
        The (ky, kx) = (0, 0) block (barotropic zero mode) is zeroed.

        The construction path is chosen in this order:

        1. Layer-sharded across ``a_shard_devices`` (when the layer count
           divides evenly by the device count).
        2. Analytic 2-layer inverse (pyqg formulation).
        3. Host-side dense inverse on Metal (no complex solve).
        4. Dense ``jnp.linalg.inv`` batched over wavenumbers.
        """
        devices = self._a_shard_devices
        if devices is not None and len(devices) > 1:
            if self.n_layers % len(devices) == 0:
                self.A = self._build_sharded_A(devices)
                return
            print(
                f"Warning: n_layers={self.n_layers} not divisible by "
                f"{len(devices)} a_shard_devices; building dense A."
            )

        if self.n_layers == 2:
            det_inv = (self.wv2 * (self.wv2 + self.F_1 + self.F_2)) ** -1.0
            det_inv = det_inv.at[0, 0].set(0.0)  # Handle zero mode

            A = jnp.zeros((2, 2, self.n_ky, self.n_kx), dtype=COMPLEX)
            A = A.at[0, 0].set(-self.wv2 - self.F_2)
            A = A.at[0, 1].set(-self.F_1)
            A = A.at[1, 0].set(-self.F_2)
            A = A.at[1, 1].set(-self.wv2 - self.F_1)
            self.A = det_inv * A
        elif selected_platform() == "metal":
            # Metal currently lacks the complex solve used by jnp.linalg.inv.
            self.A = self._build_dense_A_host()
        else:
            B = self._build_B()
            A = jnp.linalg.inv(B.transpose(2, 3, 0, 1)).transpose(2, 3, 0, 1)
            self.A = A.at[:, :, 0, 0].set(0.0)

    @property
    def B(self):
        """Matrix B satisfying qh = B ph.

        Built lazily on first access (dense, default device) and cached:
        B is dead memory in the integration path (only `_qh_from_ph` uses
        it), so it is not materialised at construction time.
        """
        if self._B_cache is None:
            self._B_cache = self._build_B()
        return self._B_cache

    @B.setter
    def B(self, value):
        self._B_cache = value

    def _ph_from_qh(self, qh):
        """Compute ph from qh using matrix inversion.

        Parameters
        ----------
        qh : array_like, shape (n_layers, n_ky, n_kx)
            Potential vorticity in Fourier space

        Returns
        -------
        ph : array_like, shape (n_layers, n_ky, n_kx)
            Streamfunction in Fourier space
        """
        ph = jnp.einsum("ijkl,jkl->ikl", self.A, qh)
        if self._a_state_sharding is not None and _under_jax_trace():
            # Help GSPMD keep A (and the output) layer-sharded when A was
            # built via a_shard_devices.
            ph = jax.lax.with_sharding_constraint(ph, self._a_state_sharding)
        return ph

    def _qh_from_ph(self, ph):
        """Compute qh from ph using direct multiplication.

        Parameters
        ----------
        ph : array_like, shape (n_layers, n_ky, n_kx)
            Streamfunction in Fourier space

        Returns
        -------
        qh : array_like, shape (n_layers, n_ky, n_kx)
            Potential vorticity in Fourier space
        """
        return jnp.einsum("ijkl,jkl->ikl", self.B, ph)

    def tree_flatten(self):
        children = ()
        aux_data = {
            "n_x": self.n_x,
            "n_y": self.n_y,
            "L_x": self.L_x,
            "L_y": self.L_y,
            "r_d": self.r_d,
            "H": self.H.tolist(),  # Convert to list for serialisation
            "a_shard_devices": self._a_shard_devices,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
