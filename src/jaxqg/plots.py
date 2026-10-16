"""Plotting functions for standard plots.
"""

import jax

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
    raise ImportError(
        "matplotlib is required for jaxqg.plot_isotropic_energy_spectrum; "
        "install it with `pip install jaxqg[notebooks]` or `pip install matplotlib`"
    ) from exc

from . import diagnostics

jnp = jax.numpy
fft_lib = jnp.fft
plt.ioff()


def plot_isotropic_energy_spectrum(
    model,
    qh,
    filename=None,
    ymin=None,
    ymax=None,
    pl_scaling=None,
    height_averaged=False,
    show_wv_d=False,
    layer_labels=None,
):
    """Plot the isotropic energy spectrum.

    Parameters
    ----------
    model : Model
        Model instance
    qh : array_like, shape (n_layers, n_ky, n_kx)
        Potential vorticity in Fourier space
    filename : str, optional
        If provided, save figure to this path
    ymin, ymax : float, optional
        Y-axis limits
    pl_scaling : tuple of floats, optional
        If provided, plot power law references with these scaling factors
        (k^-3 scaling factor, k^-5/3 scaling factor)
    height_averaged : bool
        If True, plot height-averaged spectrum instead of per-layer
    show_wv_d : bool
        If True, mark the deformation wavenumber
    layer_labels : list of str, optional
        Labels for each layer. If None, uses 'Layer 0', 'Layer 1', etc.
    """
    fig, ax = plt.subplots()

    if height_averaged:
        kr, spec_iso = diagnostics.height_averaged_isotropic_EKE_spectrum(model, qh)
        ax.loglog(kr, spec_iso, "k", linewidth=2, label="Height-averaged")
    else:
        kr, spec_iso = diagnostics.isotropic_EKE_spectrum(model, qh)

        # Generate default labels if not provided
        if layer_labels is None:
            if model.grid.n_layers == 2:
                layer_labels = ["Upper", "Lower"]
            else:
                layer_labels = [f"Layer {i}" for i in range(model.grid.n_layers)]

        # Use a colormap for distinguishing layers
        colours = plt.cm.viridis(jnp.linspace(0, 0.9, model.grid.n_layers))

        for i in range(model.grid.n_layers):
            ax.loglog(
                kr,
                spec_iso[i],
                color=colours[i],
                label=layer_labels[i] if i < len(layer_labels) else f"Layer {i}",
            )

    if pl_scaling:
        ax.loglog(
            kr,
            pl_scaling[0] * kr**-3,
            "--",
            c="lightgray",
            label=r"$k^{-3}$",
            zorder=-1,
        )
        ax.loglog(
            kr,
            pl_scaling[1] * kr ** -(5 / 3),
            ":",
            c="lightgray",
            label=r"$k^{-5/3}$",
            zorder=-1,
        )

    ax.set_xlim(kr.min(), kr.max())
    ax.set_ylim(ymin, ymax)

    if show_wv_d:
        ax.vlines(
            model.grid.wv_d,
            ymin=ax.get_ylim()[0],
            ymax=ax.get_ylim()[1],
            colors="red",
            linestyles="dotted",
            label=r"$k_d$",
        )

    ax.set_xlabel(r"$k$, m$^{-1}$")
    ax.set_ylabel(r"$E(k)$, $\mathrm{m}^3\mathrm{s}^{-2}$")
    ax.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=576)
        plt.close()
