"""
Timestepping tests inspired by pyqg AB initialisation checks.
"""

import jax.numpy as jnp

from jaxqg.backend import COMPLEX
from jaxqg.timesteppers import AB3, Euler


def test_ab3_first_step_matches_forward_euler():
    stepper = AB3(dt=2.0)
    qh = jnp.ones((2, 4, 3), dtype=COMPLEX)
    rhs = (3.0 + 1.0j) * jnp.ones_like(qh)
    rhs_m1 = jnp.zeros_like(qh)
    rhs_m2 = jnp.zeros_like(qh)

    qh_new = stepper.step(qh, rhs, rhs_m1, rhs_m2, tn=0)
    expected = qh + 2.0 * rhs

    assert jnp.allclose(qh_new, expected, rtol=1e-14, atol=1e-14)


def test_ab3_second_step_matches_ab2_coefficients():
    stepper = AB3(dt=0.5)
    qh = jnp.ones((1, 3, 3), dtype=COMPLEX)
    rhs = (2.0 + 0.0j) * jnp.ones_like(qh)
    rhs_m1 = (1.0 + 0.0j) * jnp.ones_like(qh)
    rhs_m2 = jnp.zeros_like(qh)

    qh_new = stepper.step(qh, rhs, rhs_m1, rhs_m2, tn=1)
    expected = qh + 0.5 * (1.5 * rhs - 0.5 * rhs_m1)

    assert jnp.allclose(qh_new, expected, rtol=1e-14, atol=1e-14)


def test_ab3_third_and_later_steps_match_ab3_coefficients():
    stepper = AB3(dt=0.25)
    qh = jnp.ones((1, 2, 2), dtype=COMPLEX)
    rhs = (4.0 + 1.0j) * jnp.ones_like(qh)
    rhs_m1 = (2.0 - 1.0j) * jnp.ones_like(qh)
    rhs_m2 = (1.0 + 0.5j) * jnp.ones_like(qh)

    qh_new = stepper.step(qh, rhs, rhs_m1, rhs_m2, tn=5)
    expected = qh + 0.25 * (
        (23.0 / 12.0) * rhs - (16.0 / 12.0) * rhs_m1 + (5.0 / 12.0) * rhs_m2
    )

    assert jnp.allclose(qh_new, expected, rtol=1e-14, atol=1e-14)


def test_euler_step_matches_definition():
    stepper = Euler(dt=1.5)
    qh = jnp.ones((2, 3, 3), dtype=COMPLEX)
    rhs = (0.2 - 0.3j) * jnp.ones_like(qh)

    qh_new = stepper.step(
        qh, rhs, rhs_m1=jnp.zeros_like(qh), rhs_m2=jnp.zeros_like(qh), tn=10
    )
    expected = qh + 1.5 * rhs

    assert jnp.allclose(qh_new, expected, rtol=1e-14, atol=1e-14)
