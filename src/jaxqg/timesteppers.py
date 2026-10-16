"""
Timestepping schemes to serve as attributes of a `Model` instance.
"""

import jax
from jax.tree_util import register_pytree_node_class

jnp = jax.numpy


@register_pytree_node_class
class AB3:
    """Third-order Adams-Bashforth timestepper with initialising steps."""

    def __init__(self, dt):
        self.dt = dt

    def step(self, qh, rhs, rhs_m1, rhs_m2, tn):
        a = jax.lax.select(
            tn == 0,
            jnp.array((1.0, 0, 0)),
            jax.lax.select(
                tn == 1,
                jnp.array((1.5, -0.5, 0)),
                jnp.array((23 / 12, -16 / 12, 5 / 12)),
            ),
        )
        return qh + (a[0] * rhs + a[1] * rhs_m1 + a[2] * rhs_m2) * self.dt

    def tree_flatten(self):
        children = ()
        aux_data = {"dt": self.dt}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


@register_pytree_node_class
class Euler:
    """Euler timestepper."""

    def __init__(self, dt):
        self.dt = dt

    def _euler_step(self, qh, rhs):
        """One step of forward Euler."""
        return qh + rhs * self.dt

    def step(self, qh, rhs, rhs_m1, rhs_m2, tn):
        return self._euler_step(qh, rhs)

    def tree_flatten(self):
        children = ()
        aux_data = {"dt": self.dt}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
