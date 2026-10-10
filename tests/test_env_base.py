import jax
import jax.numpy as jnp
import numpy as np
import pytest

from example.cheetah import Cheetah


@pytest.fixture(params=['np', 'jnp'])
def env(request, np_env, jnp_env):
    return {'np': np_env, 'jnp': jnp_env}[request.param]


@pytest.fixture(scope='module')
def jitted(jnp_env):
    return jax.jit(jnp_env.reset), jax.jit(jnp_env.step)


def test_invalid_backend_raises(cheetah_config):
    from minimal_mjx.utils.config import create_config_dict
    with pytest.raises(ValueError, match='Unsupported backend'):
        Cheetah(env_params=create_config_dict(cheetah_config.env_config), backend='torch')


def test_backend_modules(np_env, jnp_env):
    assert np_env._np is np and np_env.backend == 'np'
    assert jnp_env._np is jnp and jnp_env.backend == 'jnp'


def test_model_dimensions(env):
    assert (env.nq, env.nv, env.nu) == (env.mj_model.nq, env.mj_model.nv, env.mj_model.nu)
    assert env.mj_model.opt.timestep == env.sim_dt


def test_np_reset_step(np_env):
    state = np_env.reset(None)
    assert state.obs['state'].shape == (np_env.observation_size,)
    t0 = state.data.time
    state = np_env.step(state, np.zeros(np_env.action_size))
    assert state.data.time == pytest.approx(t0 + np_env.dt)
    assert set(state.metrics) == {f'reward/{k}' for k in np_env.params.reward.weights}


def test_jnp_reset_step(jnp_env, jitted):
    reset, step = jitted
    state = reset(jax.random.PRNGKey(0))
    assert state.obs['state'].shape == jnp_env.observation_size['state']
    t0 = state.data.time
    state = step(state, jnp.zeros(jnp_env.action_size))
    assert float(state.data.time) == pytest.approx(float(t0) + jnp_env.dt, abs=1e-6)


def test_backend_parity(np_env, jitted):
    """The same action sequence gives the same trajectory on both backends."""
    reset, step = jitted
    s_np = np_env.reset(None)
    s_jx = reset(jax.random.PRNGKey(0))
    actions = np.random.default_rng(0).uniform(-1, 1, size=(20, np_env.action_size))
    for a in actions:
        s_np = np_env.step(s_np, a)
        s_jx = step(s_jx, jnp.asarray(a))
        np.testing.assert_allclose(s_np.data.qpos, s_jx.data.qpos, atol=1e-4)
        np.testing.assert_allclose(s_np.data.qvel, s_jx.data.qvel, atol=1e-3)
        np.testing.assert_allclose(s_np.reward, s_jx.reward, rtol=1e-4, atol=1e-4)
        for k in s_np.metrics:
            np.testing.assert_allclose(s_np.metrics[k], s_jx.metrics[k], rtol=1e-4, atol=1e-4)


def test_np_step_does_not_mutate_input(np_env):
    state = np_env.reset(None)
    qpos, time = state.data.qpos.copy(), state.data.time
    new = np_env._step_fn(state.data, np.ones(np_env.nu))
    assert new is not state.data
    np.testing.assert_array_equal(state.data.qpos, qpos)
    assert state.data.time == time


def test_set_val_fn(env):
    arr = env._np.zeros(5)
    out = env._set_val_fn(arr, 1.0, min_idx=1, max_idx=3)
    np.testing.assert_array_equal(out, [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(arr, np.zeros(5))  # not mutated


def test_add_random_joint_state_within_limits(env):
    lo, hi = env._jt_lims
    qpos = env._np.zeros(env.nq)
    key = jax.random.PRNGKey(0)
    for _ in range(20):
        key, sub = jax.random.split(key)
        out = np.asarray(env.add_random_joint_state(sub, qpos, minval=-10.0, maxval=10.0))
        np.testing.assert_array_equal(out[:env.qpos_free], 0.0)
        assert np.all(out[env.qpos_free:] >= lo - 1e-6)
        assert np.all(out[env.qpos_free:] <= hi + 1e-6)


def test_history(env):
    x0 = env._np.array([1.0, 2.0])
    hist = env.make_history(x0, 3)
    assert hist.shape == (3, 2)
    hist = env.update_history(hist, env._np.array([9.0, 9.0]))
    np.testing.assert_array_equal(hist, [[9, 9], [1, 2], [1, 2]])


def test_get_reward_and_metrics(env):
    weights = env.params.reward.weights
    rewards = {k: float(i + 1) for i, k in enumerate(weights)}
    reward, metrics = env.get_reward_and_metrics(rewards, {})
    assert reward == pytest.approx(sum(weights[k] * v for k, v in rewards.items()))
    assert metrics == {f'reward/{k}': weights[k] * v for k, v in rewards.items()}


def test_reward_helpers(env):
    assert float(env.exp_reward(0.0, 1.0)) == pytest.approx(1.0)
    assert float(env.exp_reward(2.0, 2.0)) == pytest.approx(np.exp(-1.0))

    a = env._np.ones(3)
    assert float(env.reward_action_rate(a, a, a)) == pytest.approx(0.0)
    # act - last = 1 each (norm sqrt(3)); act - 2 last + last_last = 1 each (norm sqrt(3))
    assert float(env.reward_action_rate(a, 0 * a, 0 * a)) == pytest.approx(2 * np.sqrt(3))

    assert float(env.reward_vector_size(env._np.zeros(3), 1.0)) == pytest.approx(1.0)
    ref = env._np.array([3.0, 4.0])
    assert float(env.reward_euclidean_imitation(env._np.zeros(2), ref, 25.0)) == pytest.approx(np.exp(-1.0))
