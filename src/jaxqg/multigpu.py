"""Multi-device simulation helpers."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .model import Model


@dataclass
class MultiGPUConfig:
    """Device selection and layer-sharding options."""

    device_indices: list[int] | None = None
    """Explicit GPU indices to use. If None, uses all available GPUs."""

    prefer_gpu: bool = True
    """Prefer GPU devices over CPU."""

    layer_sharding: bool = True
    """Shard model layers across devices."""


class MultiGPUSimulator:
    """Run a QG model on selected JAX devices."""

    def __init__(self, model: Model, config: MultiGPUConfig | None = None):
        """Create a simulator for ``model`` using ``config``."""
        self.model = model
        self.config = config or MultiGPUConfig()
        self._devices = self._select_devices()

    def _select_devices(self) -> list:
        """Select and validate target devices."""
        from .backend import select_devices

        devices = select_devices(
            device_indices=self.config.device_indices,
            prefer_gpu=self.config.prefer_gpu,
        )

        return devices

    @property
    def devices(self) -> list:
        """Get list of target devices."""
        return self._devices

    @property
    def n_devices(self) -> int:
        """Number of target devices."""
        return len(self._devices)

    @property
    def device_info(self) -> dict:
        """Get device information."""
        return {
            "n_devices": self.n_devices,
            "device_indices": [int(d.id) for d in self._devices],
            "device_types": [d.device_kind for d in self._devices],
        }

    def shard_state(self, state):
        """Place ``state`` on the simulator devices, sharding layers if enabled."""
        if not self.config.layer_sharding or self.n_devices == 1:
            return jax.device_put(state, self._devices[0])

        # Check layer-device compatibility
        n_layers = self.model.grid.n_layers
        if n_layers % self.n_devices != 0:
            raise ValueError(
                f"Cannot shard {n_layers} layers across {self.n_devices} devices; "
                "the layer count must be divisible by the device count."
            )

        from .utils_multi_gpu import shard_state_across_layers

        return shard_state_across_layers(state, devices=self._devices)

    def print_info(self):
        """Print the selected devices and sharding configuration."""
        print("\n" + "=" * 70)
        print("MULTI-GPU SIMULATOR CONFIGURATION")
        print("=" * 70)
        info = self.device_info
        print(f"Number of devices:     {info['n_devices']}")
        print(f"Device indices:        {info['device_indices']}")
        print(f"Device types:          {', '.join(info['device_types'])}")
        print(f"Layer sharding:        {self.config.layer_sharding}")
        print(f"Model layers:          {self.model.grid.n_layers}")

        n_layers = self.model.grid.n_layers
        if self.config.layer_sharding and n_layers % self.n_devices == 0:
            layers_per_device = n_layers // self.n_devices
            print(f"Layers per device:     {layers_per_device}")
        print("=" * 70 + "\n")

    def run(self, state, n_steps: int, sample_rate: int = 1):
        """Integrate from ``state`` and return the trajectory and run statistics."""
        return run_simulation(self, state, n_steps, sample_rate)


@dataclass
class SimulationStats:
    """Elapsed integration time and final-state summary."""

    n_steps: int
    duration_sec: float
    steps_per_sec: float
    final_state: dict

    def __str__(self) -> str:
        return (
            f"SimulationStats(n_steps={self.n_steps}, "
            f"duration={self.duration_sec:.2f}s, "
            f"rate={self.steps_per_sec:.2f} steps/s)"
        )


def run_simulation(
    sim: MultiGPUSimulator,
    state,
    n_steps: int,
    sample_rate: int = 1,
) -> tuple[object, SimulationStats]:
    """Integrate ``state`` with ``sim`` and return the trajectory and statistics.

    The trajectory is the sampled-``State`` NamedTuple returned by
    ``Model.generate_integration_fn`` (or ``(final_state, trajectory)``
    when the model was built with ``return_final_state=True``), not a raw
    array.
    """
    import time

    # Shard state
    state_sharded = sim.shard_state(state)

    # Generate integration function
    integrate = sim.model.generate_integration_fn(
        n_steps=n_steps,
        sample_rate=sample_rate,
    )

    # Run integration
    start = time.perf_counter()
    traj = integrate(state_sharded)
    jax.block_until_ready(traj)
    elapsed = time.perf_counter() - start

    # Compile stats
    stats = SimulationStats(
        n_steps=n_steps,
        duration_sec=elapsed,
        steps_per_sec=n_steps / elapsed,
        final_state={
            "qh_shape": traj.qh[-1].shape,
            "qh_mean": float(jnp.mean(jnp.abs(traj.qh[-1]))),
            "qh_max": float(jnp.max(jnp.abs(traj.qh[-1]))),
        },
    )

    return traj, stats


def setup_multi_gpu(
    model: Model,
    device_indices: list[int] | None = None,
    verbose: bool = True,
) -> MultiGPUSimulator:
    """Create a simulator for ``model`` using the selected devices."""
    config = MultiGPUConfig(device_indices=device_indices)
    sim = MultiGPUSimulator(model, config)

    if verbose:
        sim.print_info()

    return sim
