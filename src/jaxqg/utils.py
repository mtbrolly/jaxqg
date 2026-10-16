"""Utilities.
"""

import jax

from .backend import FLOAT

jnp = jax.numpy


def _normalize_isotropic_shells(fh, spectrum, grid):
    """Remove finite-sample angular power bias while preserving radial power."""
    shell = jnp.rint(grid.wv / grid.dk_x).astype(jnp.int32)
    n_shells = int(jnp.max(shell)) + 1
    normalised = []
    for layer in range(fh.shape[0]):
        shell_power = jnp.zeros(n_shells, dtype=FLOAT)
        shell_target = jnp.zeros(n_shells, dtype=FLOAT)
        shell_power = shell_power.at[shell].add(jnp.abs(fh[layer]) ** 2)
        shell_target = shell_target.at[shell].add(0.5 * spectrum[layer])
        factor = jnp.sqrt(jnp.where(shell_power > 0.0, shell_target / shell_power, 0.0))
        normalised.append(fh[layer] * factor[shell])
    return jnp.stack(normalised)


def spectral_variance(model, fh):
    """Calculate spectral variance from real fft'ed field, fh.

    Note since we use real fft we have to nominally double the contribution
    from most modes with the exception of those in the first and last column of
    fh.
    """

    var_dens = 2.0 * jnp.abs(fh) ** 2
    var_dens = var_dens.at[..., 0].set(var_dens[..., 0] / 2)
    var_dens = var_dens.at[..., -1].set(var_dens[..., -1] / 2)
    normalisation = float(model.grid.n_x * model.grid.n_y) ** 2
    return var_dens.sum(axis=(-1, -2), dtype=FLOAT) / normalisation


def gaussian_random_field(model, spectrum, seed=1):
    """Generate an ``rfft2`` Gaussian field with the requested power spectrum.

    ``spectrum`` may have shape ``(n_ky, n_kx)`` or
    ``(n_layers, n_ky, n_kx)``. An integer ``seed`` shares a random stream
    across layers; a sequence supplies one seed per layer.
    """
    # Check if spectrum is per-layer or uniform
    if spectrum.ndim == 2:
        # Same spectrum for all layers - broadcast it
        spectrum = jnp.tile(spectrum[None, :, :], (model.grid.n_layers, 1, 1))

    # Check if seed is per-layer or uniform
    try:
        seeds = list(seed)  # If seed is iterable, use per-layer seeds
        if len(seeds) != model.grid.n_layers:
            raise ValueError(
                f"Number of seeds ({len(seeds)}) must match number of layers ({model.grid.n_layers})"
            )
    except TypeError:
        # Single seed - use same random sequence for all layers
        seeds = [seed] * model.grid.n_layers
        # Generate real-space white noise first. This guarantees Hermitian
        # consistency in both Fourier directions for the rfft2 layout.
        key = jax.random.PRNGKey(seed)
        white = jax.random.normal(key, (model.grid.n_layers,) + model.grid.x.shape)
        norm = jnp.sqrt(2.0 * model.grid.n_x * model.grid.n_y)
        spectrum = jnp.where(model.grid.wv2 > 0.0, spectrum, 0.0)
        fh = jnp.fft.rfft2(white) / norm * jnp.sqrt(spectrum)
        fh = _normalize_isotropic_shells(fh, spectrum, model.grid)
        fh = fh.at[:, 0, 0].set(0.0)
        return fh

    # Generate each layer with its own seed (for consistency with jaxqg)
    fh_layers = []
    for i, layer_seed in enumerate(seeds):
        key = jax.random.PRNGKey(layer_seed)
        white = jax.random.normal(key, model.grid.x.shape)
        norm = jnp.sqrt(2.0 * model.grid.n_x * model.grid.n_y)
        layer_spectrum = jnp.where(model.grid.wv2 > 0.0, spectrum[i], 0.0)
        fh_layer = jnp.fft.rfft2(white) / norm * jnp.sqrt(layer_spectrum)
        fh_layer = _normalize_isotropic_shells(
            fh_layer[None], layer_spectrum[None], model.grid
        )[0]
        fh_layer = fh_layer.at[0, 0].set(0.0)
        fh_layers.append(fh_layer)

    fh = jnp.stack(fh_layers)
    return fh
