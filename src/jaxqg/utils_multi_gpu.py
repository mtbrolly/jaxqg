"""
Multi-GPU layer sharding utilities.

Provides simple functions to parallelize a single simulation across multiple GPUs
by distributing layers across devices.
"""

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .model import State


def shard_state_across_layers(state, devices=None):
    """
    Distribute model state across devices by sharding along the layer dimension.

    Each GPU handles a subset of layers. JAX automatically handles communication
    during layer-coupling operations (inversion).

    Parameters
    ----------
    state : State
        Model state to be sharded
    devices : list of Device, optional
        Target devices. If None, uses all available devices from jax.devices()

    Returns
    -------
    sharded_state : State
        State with arrays distributed across devices

    Example
    -------
    >>> from jaxqg import define_state, shard_state_across_layers
    >>>
    >>> state = define_state(model, qh)
    >>> sharded_state = shard_state_across_layers(state)
    >>> trajectory = integrate(sharded_state)
    """
    if devices is None:
        devices = jax.devices()

    mesh = Mesh(np.array(devices), axis_names=("layers",))

    # Shard along layer dimension (axis 0), replicate spatial dimensions
    layer_sharding = NamedSharding(mesh, P("layers", None, None))
    replicated_sharding = NamedSharding(mesh, P())

    # Create sharded state
    sharded_state = State(
        qh=jax.device_put(state.qh, layer_sharding),
        rhs_m2=jax.device_put(state.rhs_m2, layer_sharding),
        rhs_m1=jax.device_put(state.rhs_m1, layer_sharding),
        random_key=jax.device_put(state.random_key, replicated_sharding),
        tn=state.tn,
        Mh=state.Mh if state.Mh is None else jax.device_put(state.Mh, layer_sharding),
        trainable_variables=state.trainable_variables,
        non_trainable_variables=state.non_trainable_variables,
        Rh=state.Rh if state.Rh is None else jax.device_put(state.Rh, layer_sharding),
        aux_trainable_variables=state.aux_trainable_variables,
        aux_non_trainable_variables=state.aux_non_trainable_variables,
    )

    return sharded_state
