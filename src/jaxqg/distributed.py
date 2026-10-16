"""Experimental SLURM multi-node initialisation, trialled on a single SLURM cluster.

Call before JAX backend initialisation. This path is not covered by CI or
validated on other clusters.
"""

import os
import socket
import subprocess

import jax

__all__ = ["initialize_on_slurm", "slurm_coordinator_address"]


def _hostnames():
    nodelist = os.environ["SLURM_JOB_NODELIST"]
    return (
        subprocess.check_output(["scontrol", "show", "hostnames", nodelist])
        .decode()
        .split()
    )


def slurm_coordinator_address():
    """Resolve rank-0 host's IP. The coordination service is a lightweight
    rendezvous, so the hostname IP suffices; NCCL collective traffic is
    steered onto the fast interconnect separately (via NCCL_* env)."""
    hosts = _hostnames()
    coord_host = hosts[0]
    try:
        infos = socket.getaddrinfo(coord_host, None)
    except socket.gaierror:
        return coord_host
    addrs = [inf[4][0] for inf in infos if inf[0] == socket.AF_INET]
    return addrs[0] if addrs else coord_host


def _local_gpu_count():
    """Number of GPUs visible to this process, via CUDA_VISIBLE_DEVICES or
    nvidia-smi (NOT via jax, to avoid triggering backend init before
    jax.distributed.initialize)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.strip():
        return len([p for p in cvd.split(",") if p.strip()])
    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return len([ln for ln in out.splitlines() if ln.strip()])
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return 1


def initialize_on_slurm(port=None):
    """Initialise distributed JAX from SLURM variables before backend use.

    Return 0 outside SLURM, -1 for a single-task job, or the global device
    count after initialising a multi-task job. ``port`` overrides the
    ``JAX_DIST_PORT`` environment variable (default 29400).
    """
    if "SLURM_JOB_NODELIST" not in os.environ or "SLURM_NTASKS" not in os.environ:
        return 0

    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])

    if world <= 1:
        # Single process: no distributed initialisation needed/possible.
        return -1

    port = port or int(os.environ.get("JAX_DIST_PORT", "29400"))
    coordinator = slurm_coordinator_address()

    n_local = _local_gpu_count()
    kwargs = {
        "coordinator_address": f"{coordinator}:{port}",
        "num_processes": world,
        "process_id": rank,
    }
    if n_local >= 1:
        kwargs["local_device_ids"] = list(range(n_local))

    jax.distributed.initialize(**kwargs)

    return len(jax.devices())
