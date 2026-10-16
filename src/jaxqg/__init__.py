"""
jaxqg: N-Layer Quasigeostrophic Turbulence in JAX

A spectral quasigeostrophic model supporting arbitrary number of vertical layers
with multi-GPU parallelisation support.
"""

from .backend import COMPLEX, FLOAT, select_devices, selected_platform, x64_enabled
from .diagnostics import (
    APE,
    EKE,
    EQ,
    cfl,
    eddy_turnover_time,
    height_averaged_isotropic_EKE_spectrum,
    height_averaged_isotropic_EQ_spectrum,
    isotropic_EKE_spectrum,
    isotropic_EQ_spectrum,
    modal_amplitude,
    vertical_modes,
)
from .grid import Grid
from .mechanisms import QGDynamics, SpectralFilter
from .model import (
    Model,
    State,
    create_barotropic_model,
    create_two_layer_model,
    define_state,
)
from .timesteppers import AB3, Euler
from .utils import gaussian_random_field

try:
    from .plots import plot_isotropic_energy_spectrum
except ImportError as _exc:
    _plot_import_error = _exc

    def plot_isotropic_energy_spectrum(*args, **kwargs):
        raise _plot_import_error
from .multigpu import (
    MultiGPUConfig,
    MultiGPUSimulator,
    SimulationStats,
    run_simulation,
    setup_multi_gpu,
)
from .utils_multi_gpu import shard_state_across_layers

__version__ = "1.0.0"

__all__ = [
    "AB3",
    "APE",
    "COMPLEX",
    "EKE",
    "EQ",
    "FLOAT",
    "Euler",
    "Grid",
    "Model",
    "MultiGPUConfig",
    "MultiGPUSimulator",
    "QGDynamics",
    "SimulationStats",
    "SpectralFilter",
    "State",
    "cfl",
    "create_barotropic_model",
    "create_two_layer_model",
    "define_state",
    "eddy_turnover_time",
    "gaussian_random_field",
    "height_averaged_isotropic_EKE_spectrum",
    "height_averaged_isotropic_EQ_spectrum",
    "isotropic_EKE_spectrum",
    "isotropic_EQ_spectrum",
    "modal_amplitude",
    "plot_isotropic_energy_spectrum",
    "run_simulation",
    "select_devices",
    "selected_platform",
    "setup_multi_gpu",
    "shard_state_across_layers",
    "vertical_modes",
    "x64_enabled",
]
