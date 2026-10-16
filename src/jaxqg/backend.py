"""Precision policy and device selection for jaxqg.

Select a JAX platform before process startup with ``JAX_PLATFORMS``. The
package enables x64 by default for CPU/CUDA and disables it for Metal. Set
``JAX_ENABLE_X64`` explicitly to override the default where supported.
"""

import os
from operator import index as integer_index

import jax
import jax.numpy as jnp


def _selected_platform():
    """Return the explicitly selected or active default JAX platform."""
    platforms = (
        os.environ.get("JAX_PLATFORMS") or os.environ.get("JAX_PLATFORM_NAME") or ""
    )
    first = platforms.split(",", 1)[0].strip().lower()
    if not first:
        try:
            first = jax.default_backend().lower()
        except RuntimeError:
            return None
    if first in {"metal", "mps"}:
        return "metal"
    if first in {"gpu", "cuda"}:
        return "cuda"
    return first or None


def selected_platform():
    """Return the explicitly selected or active default JAX platform."""
    return _selected_platform()


def _initialize_precision():
    platform = _selected_platform()
    requested = os.environ.get("JAX_ENABLE_X64")
    if requested is None:
        x64 = platform != "metal"
    else:
        value = requested.strip().lower()
        if value not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
            raise ValueError(
                "JAX_ENABLE_X64 must be a boolean value (0/1, true/false)."
            )
        x64 = value in {"1", "true", "yes", "on"}
    if platform == "metal" and x64:
        raise RuntimeError(
            "JAX_ENABLE_X64 is enabled, but the Metal backend does not support "
            "64-bit arrays. Set JAX_ENABLE_X64=0 before starting Python."
        )
    jax.config.update("jax_enable_x64", x64)
    return x64


_X64_ENABLED = _initialize_precision()
FLOAT = jnp.float64 if _X64_ENABLED else jnp.float32
COMPLEX = jnp.complex128 if _X64_ENABLED else jnp.complex64


def x64_enabled():
    """Whether jaxqg's process-wide precision policy enables x64."""
    return _X64_ENABLED


def select_devices(device_indices=None, prefer_gpu=True):
    """Return devices, preserving explicit index order.

    Explicit indices refer to the order returned by ``jax.devices()``. An
    invalid or unavailable index raises ``ValueError`` instead of silently
    selecting a different device set.
    """
    all_devices = jax.devices()
    if device_indices is not None:
        try:
            indices = [
                integer_index(i) if not isinstance(i, bool) else -1
                for i in device_indices
            ]
        except TypeError as exc:
            raise ValueError("device_indices must contain integers") from exc
        if not indices:
            raise ValueError("device_indices must select at least one device")
        invalid = [i for i in indices if i < 0 or i >= len(all_devices)]
        if invalid:
            raise ValueError(
                f"Unavailable device indices {invalid}; "
                + (
                    f"valid indices are 0 through {len(all_devices) - 1}."
                    if all_devices
                    else "no devices are available."
                )
            )
        if len(set(indices)) != len(indices):
            raise ValueError("device_indices must not contain duplicates")
        return [all_devices[i] for i in indices]

    if prefer_gpu:
        for kind in ("gpu", "metal", "mps"):
            try:
                found = jax.devices(kind)
            except (RuntimeError, ValueError):
                continue
            if found:
                return found
    return all_devices


__all__ = [
    "COMPLEX",
    "FLOAT",
    "select_devices",
    "selected_platform",
    "x64_enabled",
]
