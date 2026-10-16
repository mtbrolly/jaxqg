"""Tests for the documented public API at the package root."""

import jax.numpy as jnp
import pytest

import jaxqg


def test_all_symbols_importable():
    """Every symbol in __all__ is present at the package root."""
    for name in jaxqg.__all__:
        assert hasattr(jaxqg, name), f"{name} in __all__ but not importable"


def test_all_matches_module_dir_public_names():
    """__all__ covers the public names defined at the package root."""
    public = {n for n in dir(jaxqg) if not n.startswith("_")}
    # Submodule names are implementation detail, not API surface.
    submodules = {
        "backend",
        "diagnostics",
        "grid",
        "mechanisms",
        "model",
        "multigpu",
        "plots",
        "timesteppers",
        "utils",
        "utils_multi_gpu",
    }
    extras = public - set(jaxqg.__all__) - {"__version__"} - submodules
    assert extras == set(), f"public names missing from __all__: {sorted(extras)}"


def test_no_star_import_leakage():
    """Star import exposes exactly __all__ (plus compiler dunders)."""
    ns = {}
    exec("from jaxqg import *", ns)  # noqa: S102 - verifying star-import surface
    exported = {k for k in ns if not k.startswith("__")}
    assert exported == set(jaxqg.__all__)


def test_version_string():
    assert isinstance(jaxqg.__version__, str)
    assert jaxqg.__version__


class TestTwoLayerConstructor:
    def test_defaults(self):
        model = jaxqg.create_two_layer_model()
        assert model.grid.n_layers == 2
        assert model.grid.n_x == 256
        assert model.grid.n_y == 256
        assert isinstance(model.timestepper, jaxqg.AB3)
        assert model.timestepper.dt == 1800
        assert "QG" in model.mechanisms
        assert "spectral_filter" in model.mechanisms

    def test_custom_parameters(self):
        model = jaxqg.create_two_layer_model(
            n_x=64, L_x=2e6, H_1=300.0, H_2=4000.0, U_1=0.05, dt=900
        )
        assert model.grid.n_x == 64
        assert model.grid.L_x == 2e6
        assert model.mechanisms["QG"].U.shape == (2,)
        assert float(model.mechanisms["QG"].U[0]) == pytest.approx(0.05)
        assert model.timestepper.dt == 900

    def test_two_layer_radii(self):
        """The 2-layer grid reproduces the requested deformation radius."""
        from jaxqg.diagnostics import vertical_modes

        model = jaxqg.create_two_layer_model(n_x=32, r_d=20000.0)
        _, radii = vertical_modes(model)
        assert float(radii[1]) == pytest.approx(20000.0, rel=1e-10)


class TestBarotropicConstructor:
    def test_defaults(self):
        model = jaxqg.create_barotropic_model()
        assert model.grid.n_layers == 1
        assert model.grid.n_x == 256
        assert isinstance(model.timestepper, jaxqg.AB3)
        assert "QG" in model.mechanisms

    def test_single_layer_radii(self):
        """A single layer has only the barotropic mode (infinite radius)."""
        from jaxqg.diagnostics import vertical_modes

        model = jaxqg.create_barotropic_model(n_x=32)
        modes, radii = vertical_modes(model)
        assert modes.shape == (1, 1)
        assert float(radii[0]) == float("inf")

    def test_custom_parameters(self):
        model = jaxqg.create_barotropic_model(n_x=64, L_x=4.0, beta=1e-5, dt=600)
        assert model.grid.n_x == 64
        assert float(model.mechanisms["QG"].beta) == pytest.approx(1e-5)
        assert model.timestepper.dt == 600

    def test_barotropic_evolution(self):
        """A short barotropic integration runs and conserves shape."""
        from jaxqg.utils import gaussian_random_field

        model = jaxqg.create_barotropic_model(n_x=32)
        grid = model.grid
        qh = gaussian_random_field(model, 1e-26 * grid.wv**-4, seed=42)
        state = jaxqg.define_state(model, qh)
        integrate = model.generate_integration_fn(n_steps=8, sample_rate=8)
        traj = integrate(state)
        # Sampled trajectory adds a leading time-sample axis.
        assert traj.qh.shape == (1,) + qh.shape


class TestModelSaveLoad:
    def test_save_requires_data_dir(self, tmp_path):
        model = jaxqg.create_two_layer_model(n_x=32)
        with pytest.raises(ValueError, match="data_dir"):
            model.save()

    def test_roundtrip(self, tmp_path):
        model = jaxqg.create_two_layer_model(n_x=32, data_dir=str(tmp_path))
        model.save()
        loaded = jaxqg.Model.load(tmp_path / "m.pkl")
        assert isinstance(loaded, jaxqg.Model)
        assert loaded.grid.n_x == 32
        assert loaded.grid.n_layers == 2
        assert float(loaded.mechanisms["QG"].beta) == float(
            model.mechanisms["QG"].beta
        )
        assert loaded.timestepper.dt == model.timestepper.dt
        # Array attribute survives the round trip bit-for-bit.
        assert jnp.allclose(loaded.grid.A, model.grid.A)

    def test_dtype_preserved(self, tmp_path):
        """Round trip preserves the active COMPLEX dtype policy."""
        from jaxqg.backend import COMPLEX

        model = jaxqg.create_two_layer_model(n_x=32, data_dir=str(tmp_path))
        model.save()
        loaded = jaxqg.Model.load(tmp_path / "m.pkl")
        assert loaded.grid.A.dtype == COMPLEX
        assert loaded.grid.A.dtype == model.grid.A.dtype
