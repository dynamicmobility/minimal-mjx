"Create default, single or parallel instance 'default policies."

import jax
import jax.numpy as jnp
from minimal_mjx.utils.state import get_batch_shape

def make_zero_policy(action_size: int):
    """Open-loop policy that always emits zero actions."""

    def policy(obs, key, t):
        batch_shape = get_batch_shape(obs)
        return jnp.zeros((*batch_shape, action_size)), {}

    return policy


def make_random_policy(action_size: int):
    """Open-loop policy that emits uniform random actions in ``[-1, 1]``."""

    def policy(obs, key, t):
        batch_shape = get_batch_shape(obs)
        action = jax.random.uniform(
            key       = key,
            shape     = (*batch_shape, action_size),
            minval    = -1.0,
            maxval    = 1.0
        )
        return action, {}

    return policy


def make_sinusoidal_policy(
    action_size: int, amp: float = 0.8, freq: float = 1.5, phases=None
):
    """Open-loop sinusoid ``amp * sin(2*pi*freq*t + phases)``, identical across envs.
    """
    if phases is None:
        phases = jnp.zeros(action_size)
    else:
        phases = jnp.asarray(phases)

    def policy(obs, key, t):
        batch_shape = get_batch_shape(obs)
        action = amp * jnp.sin(2.0 * jnp.pi * freq * t + phases)  # (action_size,)
        return jnp.broadcast_to(action, (*batch_shape, action_size)), {}

    return policy


def from_inference_fn(base_policy):
    """Adapt a normal brax ``policy(obs, key)`` to a time-based policy
    ``(obs, key, t)`` protocol.
    """

    def policy(obs, key, t): # t is unused
        return base_policy(obs, key)

    return policy


OPEN_LOOP = {
    "zero": make_zero_policy,
    "random": make_random_policy,
    "sinusoid": make_sinusoidal_policy,
}


def make_open_loop_policy(policy_desc: str, action_size: int, **kwargs):
    """Build an open-loop policy by name (``'zero'``, ``'random'``, ``'sinusoid'``).

    Extra kwargs are forwarded to the underlying factory.
    """
    if policy_desc not in OPEN_LOOP:
        raise ValueError(
            f"Unknown open-loop policy '{policy_desc}'. Expected one of {list(OPEN_LOOP)}."
        )
    factory = OPEN_LOOP[policy_desc]
    return factory(action_size, **kwargs)
