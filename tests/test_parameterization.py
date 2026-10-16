"""
Parameterisation and mechanism-invariant tests inspired by pyqg behaviour tests.
"""

import jax
import jax.numpy as jnp

from jaxqg import Model, define_state
from jaxqg.backend import COMPLEX
from jaxqg.mechanisms import QGDynamics, SpectralFilter
from jaxqg.timesteppers import AB3


def _make_base_model(grid):
    timestepper = AB3(dt=1800)
    mechanisms = {
        "QG": QGDynamics(grid, U=jnp.zeros(grid.n_layers), beta=0.0, gamma=0.0),
        "filter": SpectralFilter(grid),
    }
    return Model(grid, mechanisms, timestepper)


def test_q_parameterization_adds_increment_exactly(grid_2layer):
    """Single-step difference should equal parameterisation increment Mh."""
    base_model = _make_base_model(grid_2layer)

    qh0 = jnp.zeros(
        (grid_2layer.n_layers, grid_2layer.n_ky, grid_2layer.n_kx), dtype=COMPLEX
    )
    qh0 = qh0.at[:, 1, 1].set(1e-6 + 0j)

    state_base = define_state(base_model, qh0, seed=1)

    def param_func(trainable_vars, non_trainable_vars, qh, Mh, Rh, key):
        amp = trainable_vars["amp"]
        return amp * jnp.ones_like(qh), non_trainable_vars

    model_param = Model(
        base_model.grid,
        base_model.mechanisms,
        base_model.timestepper,
        param_func=param_func,
    )

    trainable = {"amp": jnp.array(2.5e-9 + 0j, dtype=COMPLEX)}
    state_param = define_state(
        model_param,
        qh0,
        seed=1,
        trainable_variables=trainable,
        non_trainable_variables={},
    )

    out_base = base_model.evolve_one_step_stateless(None, state_base)
    out_param = model_param.evolve_one_step_stateless(None, state_param)

    # Mh is added before exact mechanisms; with the default spectral filter,
    # the net increment is filtered Mh.
    expected_delta = trainable["amp"] * model_param.mechanisms["filter"].exp_filter
    expected_delta = jnp.broadcast_to(expected_delta, qh0.shape)
    assert jnp.allclose(
        out_param.qh - out_base.qh, expected_delta, rtol=1e-12, atol=1e-12
    )


def test_aux_function_updates_rh_state(grid_2layer):
    """Auxiliary function should update Rh and preserve shape."""
    base_model = _make_base_model(grid_2layer)

    def aux_func(aux_trainable, aux_non_trainable, qh, Mh, Rh, key):
        updated = qh if Rh is None else Rh + qh
        return updated, aux_non_trainable

    model_aux = Model(
        base_model.grid,
        base_model.mechanisms,
        base_model.timestepper,
        aux_func=aux_func,
    )

    qh0 = jnp.zeros(
        (grid_2layer.n_layers, grid_2layer.n_ky, grid_2layer.n_kx), dtype=COMPLEX
    )
    qh0 = qh0.at[:, 2, 2].set(3e-7 + 1e-7j)
    state0 = define_state(
        model_aux,
        qh0,
        seed=2,
        aux_trainable_variables={},
        aux_non_trainable_variables={},
    )

    out1 = model_aux.evolve_one_step_stateless(None, state0)
    out2 = model_aux.evolve_one_step_stateless(None, out1)

    assert out1.Rh is not None
    assert out1.Rh.shape == qh0.shape
    assert jnp.allclose(out1.Rh, qh0, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(out2.Rh, qh0 + out1.qh, rtol=1e-10, atol=1e-10)


def test_bottom_drag_only_affects_bottom_layer_exactly(grid_2layer):
    """Difference between drag/no-drag RHS should match gamma*wavenumber term at bottom only."""
    gamma = 1e-6
    qh = jnp.zeros((2, grid_2layer.n_ky, grid_2layer.n_kx), dtype=COMPLEX)
    qh = qh.at[:, 3, 4].set(jnp.array([1.0 + 0.0j, -0.5 + 0.25j]))

    dyn_no_drag = QGDynamics(
        grid_2layer, U=jnp.zeros(2), V=jnp.zeros(2), beta=0.0, gamma=0.0
    )
    dyn_drag = QGDynamics(
        grid_2layer, U=jnp.zeros(2), V=jnp.zeros(2), beta=0.0, gamma=gamma
    )

    rhs_no_drag = dyn_no_drag._rhs(qh)
    rhs_drag = dyn_drag._rhs(qh)

    ph = grid_2layer._ph_from_qh(qh)
    expected_delta = jnp.zeros_like(qh)
    expected_delta = expected_delta.at[-1].set(gamma * grid_2layer.wv2 * ph[-1])

    assert jnp.allclose(rhs_drag - rhs_no_drag, expected_delta, rtol=1e-10, atol=1e-10)


def test_zero_mode_constant_q_has_zero_rhs_when_no_forcing(grid_2layer):
    """Constant PV (zero mode only) with U=V=beta=gamma=0 has no tendency."""
    dyn = QGDynamics(
        grid_2layer, U=jnp.zeros(2), V=jnp.zeros(2), beta=0.0, gamma=0.0
    )

    qh = jnp.zeros((2, grid_2layer.n_ky, grid_2layer.n_kx), dtype=COMPLEX)
    qh = qh.at[:, 0, 0].set(jnp.array([1.0 + 0.0j, -2.0 + 0.0j]))

    rhs = dyn._rhs(qh)

    assert jnp.allclose(rhs, 0.0, atol=1e-14)


def test_parameterization_model_jittable(grid_2layer):
    """Model with param_func should run through JIT path."""
    base_model = _make_base_model(grid_2layer)

    def param_func(trainable_vars, non_trainable_vars, qh, Mh, Rh, key):
        return trainable_vars["amp"] * jnp.ones_like(qh), non_trainable_vars

    model = Model(
        base_model.grid,
        base_model.mechanisms,
        base_model.timestepper,
        param_func=param_func,
    )

    qh0 = jnp.zeros(
        (grid_2layer.n_layers, grid_2layer.n_ky, grid_2layer.n_kx), dtype=COMPLEX
    )
    state = define_state(
        model,
        qh0,
        trainable_variables={"amp": jnp.array(1e-10 + 0j, dtype=COMPLEX)},
        non_trainable_variables={},
    )

    step = jax.jit(lambda s: model.evolve_one_step_stateless(None, s))
    out = step(state)

    assert out.qh.shape == qh0.shape
    assert out.tn == 1
