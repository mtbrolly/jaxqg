"""
QG model.
"""

import pickle
from collections import namedtuple
from functools import cache
from operator import index as integer_index
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@cache
def _sampled_trajectory_class(fields):
    """NamedTuple class for scan outputs restricted to `fields` of State."""
    return namedtuple("SampledTrajectory", fields)


@register_pytree_node_class
class Model:
    """Object to collect together a grid, a timestepper, a set of
    mechanisms and stateless methods for evolution.

    Parameterisations can be added, either Markovian (by specifying param_func)
    or extended Markovian (by specifying both param_func and aux_func).
    """

    def __init__(
        self,
        grid,
        mechanisms,
        timestepper,
        param_func=None,
        aux_func=None,
        data_dir=None,
    ):
        self.grid = grid
        self.mechanisms = mechanisms
        self.timestepper = timestepper
        self.param_func = param_func
        self.aux_func = aux_func
        self.data_dir = data_dir
        if self.data_dir:
            self._create_data_dir()

    def tree_flatten(self):
        children = ()
        aux_data = {
            "grid": self.grid,
            "mechanisms": self.mechanisms,
            "timestepper": self.timestepper,
            "param_func": self.param_func,
            "aux_func": self.aux_func,
            "data_dir": self.data_dir,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def _create_data_dir(self):
        if not Path(self.data_dir).exists():
            Path(self.data_dir).mkdir(parents=True)

    def save(self):
        """Save Model instance."""
        if self.data_dir is None:
            raise ValueError("data_dir is not set. Cannot save the model.")
        with open(Path(self.data_dir) / "m.pkl", "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file):
        """Load Model instance from file."""
        with open(file, "rb") as file_:
            return pickle.load(file_)

    def _evolve_one_step_stateless(self, state):

        (
            qh,
            rhs_m2,
            rhs_m1,
            random_key,
            tn,
            Mh,
            trainable_variables,
            non_trainable_variables,
            Rh,
            aux_trainable_variables,
            aux_non_trainable_variables,
        ) = state

        # Compute Mh
        if self.param_func:
            random_key, subkey = jax.random.split(random_key, 2)
            Mh, non_trainable_variables = self.param_func(
                trainable_variables, non_trainable_variables, qh, Mh, Rh, subkey
            )

        # Increment auxiliary variable
        if self.aux_func:
            random_key, subkey = jax.random.split(random_key, 2)
            Rh, aux_non_trainable_variables = self.aux_func(
                aux_trainable_variables, aux_non_trainable_variables, qh, Mh, Rh, subkey
            )

        # Do QG dynamics (except small-scale dissipation)
        if "QG" in self.mechanisms:
            rhs = self.mechanisms["QG"]._rhs(qh)
            qh = self.timestepper.step(qh, rhs, rhs_m1, rhs_m2, tn)
            rhs_m2 = rhs_m1
            rhs_m1 = rhs

        # Add Mh
        if self.param_func:
            qh += Mh

        # Apply the optional spectral filter.
        for filter_name in ("filter", "spectral_filter"):
            if filter_name in self.mechanisms:
                qh = self.mechanisms[filter_name](qh)

        tn += 1
        return State(
            qh=qh,
            rhs_m2=rhs_m2,
            rhs_m1=rhs_m1,
            random_key=random_key,
            tn=tn,
            Mh=Mh,
            trainable_variables=trainable_variables,
            non_trainable_variables=non_trainable_variables,
            Rh=Rh,
            aux_trainable_variables=aux_trainable_variables,
            aux_non_trainable_variables=aux_non_trainable_variables,
        )

    def evolve_one_step_stateless(self, _, state):
        return self._evolve_one_step_stateless(state)

    def generate_integration_fn(
        self,
        n_steps,
        sample_rate=None,
        unroll=1,
        sample_fields=None,
        return_final_state=False,
        checkpoint_rate=1,
    ):
        """Build a scan-based ``integrate(state)`` function.

        ``sample_rate`` must divide ``n_steps``; it controls which states are
        returned (an output-sampling knob). ``sample_fields`` limits stored
        outputs; ``return_final_state`` returns the final state with the
        sampled trajectory for chunked runs.

        ``checkpoint_rate`` (steps per ``jax.checkpoint`` boundary) is a
        memory/speed knob for autodiff: it must divide ``sample_rate``.
        ``checkpoint_rate=1`` (default) stores the least memory during the
        backward pass and measured fastest on GPU; larger values trade more
        backward-pass memory (and slightly more time on GPU) for fewer
        checkpoint boundaries. On the CPU and Metal (MPS) backends, where
        per-op launch overhead is high, set ``checkpoint_rate=8`` (and
        ``unroll=8``); the defaults target GPU. None of these knobs change
        results, only memory/runtime.

        ``unroll`` (compile-time unrolling of the per-chunk loop) is
        clamped to ``checkpoint_rate``, since the unrolled loop spans a
        single checkpointed chunk; values beyond it have no effect.
        """

        def count(value, name):
            if isinstance(value, bool):
                raise TypeError(f"{name} must be an integer, not bool")
            try:
                value = integer_index(value)
            except TypeError as exc:
                raise ValueError(f"{name} must be a positive integer") from exc
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            return value

        n_steps = count(n_steps, "n_steps")
        if sample_rate is None:
            sample_rate = n_steps
        sample_rate = count(sample_rate, "sample_rate")
        if checkpoint_rate is None:
            checkpoint_rate = 1
        checkpoint_rate = count(checkpoint_rate, "checkpoint_rate")
        unroll = count(unroll, "unroll")
        unroll = min(unroll, checkpoint_rate)
        if n_steps % sample_rate != 0:
            raise ValueError(
                f"n_steps ({n_steps}) must be divisible by sample_rate ({sample_rate})"
            )
        if sample_rate % checkpoint_rate != 0:
            raise ValueError(
                f"sample_rate ({sample_rate}) must be divisible by "
                f"checkpoint_rate ({checkpoint_rate})"
            )

        if sample_fields is None:

            def project_outputs(state):
                return state
        else:
            fields = tuple(sample_fields)
            unknown = [f for f in fields if f not in State._fields]
            if unknown:
                raise ValueError(
                    f"Unknown sample_fields {unknown}; State fields are {State._fields}"
                )
            SampledTrajectory = _sampled_trajectory_class(fields)

            def project_outputs(state):
                return SampledTrajectory(*(getattr(state, f) for f in fields))

        if checkpoint_rate == sample_rate:

            @jax.checkpoint
            def integration_interval(state, _):
                outputs = jax.lax.fori_loop(
                    0, sample_rate, self.evolve_one_step_stateless, state, unroll=unroll
                )
                return outputs, project_outputs(outputs)

        else:
            # Finer checkpoint granularity than sampling: rematerialize in
            # checkpoint_rate-step chunks inside each sampling interval, so
            # the AD tape stores only chunk-boundary states plus the
            # requested samples.
            chunks_per_sample = sample_rate // checkpoint_rate

            remat_chunk = jax.checkpoint(
                lambda state: jax.lax.fori_loop(
                    0, checkpoint_rate, self.evolve_one_step_stateless, state, unroll=unroll
                )
            )

            def integration_interval(state, _):
                outputs = jax.lax.fori_loop(
                    0, chunks_per_sample, lambda _, s: remat_chunk(s), state
                )
                return outputs, project_outputs(outputs)

        def integrate(state):
            # Pass xs=None explicitly with length= so this works on both the
            # current and older JAX releases (the length-only form was added
            # later; older arrays expect a positional xs argument).
            final_state, traj = jax.lax.scan(
                integration_interval, state, None, length=n_steps // sample_rate
            )
            return (final_state, traj) if return_final_state else traj

        return integrate


class State(NamedTuple):
    qh: jnp.ndarray  # PV anomaly in Fourier space (n_layers, n_ky, n_kx)
    rhs_m2: jnp.ndarray  # RHS from 2 steps ago
    rhs_m1: jnp.ndarray  # RHS from 1 step ago
    random_key: jnp.ndarray  # JAX random key
    tn: int  # Current time step
    Mh: jnp.ndarray  # Parameterisation (Fourier space)
    trainable_variables: dict  # Trainable params for param_func
    non_trainable_variables: dict  # Non-trainable params for param_func
    Rh: jnp.ndarray  # Auxiliary variable (Fourier space)
    aux_trainable_variables: dict  # Trainable params for aux_func
    aux_non_trainable_variables: dict  # Non-trainable params for aux_func


def define_state(
    model,
    qh,
    seed=1,
    trainable_variables=None,
    non_trainable_variables=None,
    aux_trainable_variables=None,
    aux_non_trainable_variables=None,
):
    """Initialise model state.

    Parameters
    ----------
    model : Model
        Model instance
    qh : array_like, shape (n_layers, n_ky, n_kx)
        Initial potential vorticity in Fourier space
    seed : int
        Random seed (default: 1)
    trainable_variables : dict, optional
        Trainable parameters for parameterisation
    non_trainable_variables : dict, optional
        Non-trainable parameters for parameterisation
    aux_trainable_variables : dict, optional
        Trainable parameters for auxiliary function
    aux_non_trainable_variables : dict, optional
        Non-trainable parameters for auxiliary function

    Returns
    -------
    state : State
        Initial state
    """
    if qh.shape != (model.grid.n_layers, model.grid.n_ky, model.grid.n_kx):
        raise ValueError(
            f"qh has shape {qh.shape},"
            + f" expected {(model.grid.n_layers, model.grid.n_ky, model.grid.n_kx)}"
        )
    return State(
        qh=qh,
        rhs_m2=jnp.zeros_like(qh),
        rhs_m1=jnp.zeros_like(qh),
        random_key=jax.random.PRNGKey(seed),
        tn=0,
        Mh=None,
        trainable_variables=trainable_variables,
        non_trainable_variables=non_trainable_variables,
        Rh=None,
        aux_trainable_variables=aux_trainable_variables,
        aux_non_trainable_variables=aux_non_trainable_variables,
    )


def create_two_layer_model(
    n_x=256,
    L_x=1e6,
    L_y=1e6,
    H_1=500.0,
    H_2=5000.0,
    r_d=15000.0,
    U_1=0.025,
    U_2=0.0,
    beta=1e-11,
    gamma=7e-8,
    dt=1800,
    data_dir=None,
):
    """Create a 2-layer quasi-geostrophic model.

    Parameters
    ----------
    n_x : int
        Grid resolution in x-direction (y-resolution is same)
    L_x, L_y : float
        Domain size in x and y (meters)
    H_1, H_2 : float
        Layer thicknesses (meters)
    r_d : float
        First baroclinic deformation radius (meters)
    U_1, U_2 : float
        Mean zonal flows in each layer (m/s)
    beta : float
        Coriolis parameter gradient (1/m/s)
    gamma : float
        Dissipation coefficient
    dt : float
        Timestep (seconds)
    data_dir : str, optional
        Directory for saving outputs

    Returns
    -------
    model : Model
        Initialised 2-layer model
    """
    from . import mechanisms
    from .grid import Grid
    from .timesteppers import AB3

    grid = Grid(n_x=n_x, n_y=n_x, L_x=L_x, L_y=L_y, H=[H_1, H_2], r_d=r_d)

    timestepper = AB3(dt)

    U = jnp.array([U_1, U_2])
    mechanisms_ = {
        "QG": mechanisms.QGDynamics(grid, U=U, beta=beta, gamma=gamma),
        "spectral_filter": mechanisms.SpectralFilter(grid),
    }

    return Model(grid, mechanisms_, timestepper, data_dir=data_dir)


def create_barotropic_model(
    n_x=256, L_x=2 * jnp.pi, L_y=2 * jnp.pi, beta=0.0, gamma=0.0, dt=1800, data_dir=None
):
    """Create a 1-layer barotropic turbulence model.

    Simulates 2D barotropic flow (no vertical stratification).
    Default parameters are standard for 2D turbulence on an f-plane (beta=0)
    with no dissipation (gamma=0).

    Parameters
    ----------
    n_x : int
        Grid resolution in x-direction (y-resolution is same)
    L_x, L_y : float
        Domain size in x and y (meters)
    beta : float
        Coriolis parameter gradient (1/m/s). Set to 0 for non-rotating turbulence.
    gamma : float
        Dissipation coefficient
    dt : float
        Timestep (seconds)
    data_dir : str, optional
        Directory for saving outputs

    Returns
    -------
    model : Model
        Initialised 1-layer barotropic model
    """
    from . import mechanisms
    from .grid import Grid
    from .timesteppers import AB3

    # Create single-layer grid with no stratification
    grid = Grid(n_x=n_x, n_y=n_x, L_x=L_x, L_y=L_y, H=[1000.0], r_d=1e10)

    timestepper = AB3(dt)

    U = jnp.array([0.0])
    mechanisms_ = {
        "QG": mechanisms.QGDynamics(grid, U=U, beta=beta, gamma=gamma),
        "spectral_filter": mechanisms.SpectralFilter(grid),
    }

    return Model(grid, mechanisms_, timestepper, data_dir=data_dir)
