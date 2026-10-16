"""
Mechanisms to be implemented in a `Model` instance.
"""

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .backend import FLOAT

fft_lib = jnp.fft


@register_pytree_node_class
class QGDynamics:
    """Everything but small scale dissipation. Not dealiased.
    """

    def __init__(self, grid, U=None, V=None, beta=0.0, gamma=0.0):
        """Initialise N-layer QG dynamics.

        Parameters
        ----------
        grid : Grid
            Grid object containing spatial information
        U : array_like, optional
            Zonal mean flow in each layer (shape: n_layers). If None, defaults to zeros.
            For 2-layer backward compatibility, can pass U_1, U_2 via kwargs.
        V : array_like, optional
            Meridional mean flow in each layer (shape: n_layers). If None, defaults to zeros.
            For 2-layer backward compatibility, can pass V_1, V_2 via kwargs.
        beta : float
            Planetary vorticity gradient (default: 0)
        gamma : float
            Bottom drag coefficient (default: 0)
        """
        self.grid = grid
        self._beta_vec_cache = None
        self._Qy_vec_cache = None
        self._U_vec_cache = None
        self._V_vec_cache = None

        # Handle zonal mean flow
        if U is None:
            U = jnp.zeros(grid.n_layers, dtype=FLOAT)
        self.U = jnp.array(U, dtype=FLOAT)

        # Handle meridional mean flow
        if V is None:
            V = jnp.zeros(grid.n_layers, dtype=FLOAT)
        self.V = jnp.array(V, dtype=FLOAT)

        self.beta = beta
        self.gamma = gamma

        # For backward compatibility
        if grid.n_layers == 2:
            self.U_1 = self.U[0]
            self.U_2 = self.U[1]
            self.V_1 = self.V[0]
            self.V_2 = self.V[1]

        self._initialise_base_state_forcing()

    def _initialise_base_state_forcing(self):
        """Initialise beta effect and PV gradients including contributions from mean flow shear.

        The effective beta in each layer (β_eff, the background zonal PV
        gradient) includes the planetary beta plus contributions from the
        vertical shear of the zonal mean flow.

        The meridional PV gradient (Q_y) comes from the stretching effect of
        meridional mean flow.

        For the 2-layer case:
          β_1 = β + F_1 * (U_1 - U_2)
          β_2 = β + F_2 * (U_2 - U_1)
          Q_y[1] = F_1 * (V_2 - V_1)
          Q_y[2] = F_2 * (V_1 - V_2)

        For N layers, this generalises based on interface coupling parameters.

        Note: F_interface_up[j] is the coupling from layer j to j+1
              F_interface_down[j] is the coupling from layer j+1 to j
        """
        # Effective beta in each layer (background zonal PV gradient; enters
        # _rhs as -β_eff ∂ψ/∂x).
        beta_layer = jnp.zeros(self.grid.n_layers, dtype=FLOAT)
        # Meridional PV gradient in each layer
        Qy_layer = jnp.zeros(self.grid.n_layers, dtype=FLOAT)

        for i in range(self.grid.n_layers):
            beta_i = self.beta
            qy_i = 0.0

            # Contribution from layer above (if exists)
            if i > 0:
                beta_i = beta_i + self.grid.F_interface_down[i - 1] * (
                    self.U[i] - self.U[i - 1]
                )
                # Meridional contribution: F couples layer i to i-1
                qy_i = qy_i + self.grid.F_interface_down[i - 1] * (
                    self.V[i - 1] - self.V[i]
                )

            # Contribution from layer below (if exists)
            if i < self.grid.n_layers - 1:
                beta_i = beta_i + self.grid.F_interface_up[i] * (
                    self.U[i] - self.U[i + 1]
                )
                # Meridional contribution: F couples layer i to i+1
                qy_i = qy_i + self.grid.F_interface_up[i] * (self.V[i + 1] - self.V[i])

            beta_layer = beta_layer.at[i].set(beta_i)
            Qy_layer = Qy_layer.at[i].set(qy_i)

        # Effective beta in each layer (background zonal PV gradient; used
        # with ikx in _rhs, shape: n_layers). Broadcast to
        # (n_layers, n_ky, n_kx) only through the lazy `beta_vec` property.
        self.beta_layer = beta_layer

        # Meridional PV gradient in each layer (shape: n_layers)
        self.Qy_layer = Qy_layer

        # For backward compatibility with 2-layer model
        if self.grid.n_layers == 2:
            self.beta1 = beta_layer[0]
            self.beta2 = beta_layer[1]

    def _broadcast_spectral(self, layer_values):
        """Materialize a (n_layers,) vector to (n_layers, n_ky, n_kx)."""
        return jnp.broadcast_to(
            layer_values[:, None, None],
            (self.grid.n_layers, self.grid.n_ky, self.grid.n_kx),
        )

    def _broadcast_real(self, layer_values):
        """Materialize a (n_layers,) vector to (n_layers, n_y, n_x)."""
        return jnp.broadcast_to(
            layer_values[:, None, None], (self.grid.n_layers,) + self.grid.x.shape
        )

    @property
    def beta_vec(self):
        """Effective beta broadcast to (n_layers, n_ky, n_kx) (lazy)."""
        if self._beta_vec_cache is None:
            self._beta_vec_cache = self._broadcast_spectral(self.beta_layer)
        return self._beta_vec_cache

    @property
    def Qy_vec(self):
        """Meridional PV gradient broadcast to (n_layers, n_ky, n_kx) (lazy)."""
        if self._Qy_vec_cache is None:
            self._Qy_vec_cache = self._broadcast_spectral(self.Qy_layer)
        return self._Qy_vec_cache

    @property
    def U_vec(self):
        """Zonal mean flow broadcast to (n_layers, n_y, n_x) (lazy)."""
        if self._U_vec_cache is None:
            self._U_vec_cache = self._broadcast_real(self.U)
        return self._U_vec_cache

    @property
    def V_vec(self):
        """Meridional mean flow broadcast to (n_layers, n_y, n_x) (lazy)."""
        if self._V_vec_cache is None:
            self._V_vec_cache = self._broadcast_real(self.V)
        return self._V_vec_cache

    def _rhs(self, qh):
        """Compute right-hand side of QG equations.

        For each layer i (with u = -∂ψ/∂y, v = ∂ψ/∂x the eddy velocities):
        ∂q[i]/∂t = -∂((U[i]+u) q[i])/∂x - ∂((V[i]+v) q[i])/∂y
                    - β_eff[i] ∂ψ[i]/∂x + Q_y[i] ∂ψ[i]/∂y
                    + γ ∇²ψ[N]  (bottom layer only, if gamma ≠ 0)

        where β_eff is the background zonal PV gradient (planetary beta plus
        zonal-shear contributions) and Q_y is the background meridional PV
        gradient (from meridional-shear stretching).

        Parameters
        ----------
        qh : array_like, shape (n_layers, n_ky, n_kx)
            Potential vorticity in Fourier space

        Returns
        -------
        rhs : array_like, shape (n_layers, n_ky, n_kx)
            Time tendency in Fourier space
        """
        # Get streamfunction from PV
        ph = self.grid._ph_from_qh(qh)

        # Compute velocities in Fourier space
        uh = -self.grid.iky * ph  # u = -∂ψ/∂y
        vh = self.grid.ikx * ph  # v = ∂ψ/∂x

        # Transform to real space
        q = jnp.fft.irfft2(qh)
        u = jnp.fft.irfft2(uh)
        v = jnp.fft.irfft2(vh)

        # Compute advection terms for the total mean plus perturbation flow.
        uq = (self.U[:, None, None] + u) * q
        vq = (self.V[:, None, None] + v) * q

        # Transform back to Fourier space
        uqh = jnp.fft.rfft2(uq)
        vqh = jnp.fft.rfft2(vq)

        # Advection: -∂(uq)/∂x - ∂(vq)/∂y
        advection = -(self.grid.ikx * uqh + self.grid.iky * vqh)

        # Beta effect: -β_eff * ∂ψ/∂x = -β_eff * ikx * ph (zonal PV gradient)
        beta_term = -self.beta_layer[:, None, None] * self.grid.ikx * ph

        # Meridional shear effect: -u * Q_y = Q_y * ∂ψ/∂y.
        meridional_term = self.Qy_layer[:, None, None] * self.grid.iky * ph

        # Combine terms
        rhs = advection + beta_term + meridional_term

        # Bottom drag (Ekman friction): γ * ∇²ψ, applied only to bottom layer
        if self.gamma != 0:
            bf = self.gamma * self.grid.wv2 * ph
            # Zero out all layers except the bottom layer
            bf = bf.at[:-1].set(0.0)
            rhs = rhs + bf

        return rhs

    def tree_flatten(self):
        children = ()
        aux_data = {
            "grid": self.grid,
            "U": self.U.tolist(),
            "V": self.V.tolist(),
            "beta": self.beta,
            "gamma": self.gamma,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


@register_pytree_node_class
class SpectralFilter:
    """Exponential spectral filter for applying highly scale-selective but
    non-physical dissipation at small scales.

    Works for any number of layers.
    """

    def __init__(self, grid):
        self.grid = grid
        filterfac = 23.6
        cphi = 0.65 * jnp.pi
        wvx = jnp.sqrt((grid.kx * grid.dx) ** 2.0 + (grid.ky * grid.dy) ** 2.0)
        exp_filter = jnp.exp(-filterfac * (wvx - cphi) ** 4.0)
        exp_filter = jnp.where(wvx <= cphi, 1.0, exp_filter)
        self.exp_filter = exp_filter

    def __call__(self, qh):
        """Apply filter to all layers.

        Parameters
        ----------
        qh : array_like, shape (n_layers, n_ky, n_kx)
            Potential vorticity in Fourier space

        Returns
        -------
        filtered_qh : array_like, shape (n_layers, n_ky, n_kx)
            Filtered potential vorticity
        """
        return qh * self.exp_filter

    def tree_flatten(self):
        children = ()
        aux_data = {"grid": self.grid}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
