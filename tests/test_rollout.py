import numpy as np
import pytest

import minimal_mjx as mm
from minimal_mjx.eval.rollout import make_dummy_inference_fn
from minimal_mjx.utils import plotting


def test_dummy_inference_zero(np_env):
    fn = make_dummy_inference_fn(np_env, mode='zero')
    action, extra = fn(None, None)
    np.testing.assert_array_equal(action, np.zeros(np_env.action_size))
    assert extra is None


def test_dummy_inference_random(np_env):
    fn = make_dummy_inference_fn(np_env, mode='random')
    actions = np.stack([fn(None, None)[0] for _ in range(100)])
    assert actions.shape == (100, np_env.action_size)
    assert np.all(actions >= -1) and np.all(actions <= 1)
    assert actions.std() > 0


def test_dummy_inference_unknown_mode(np_env):
    with pytest.raises(Exception, match='Unknown inference type'):
        make_dummy_inference_fn(np_env, mode='bogus')


def _rollout(env, inference_fn, n_steps, **kwargs):
    return mm.eval.rollout_policy(
        inference_fn=inference_fn, env=env, n_steps=n_steps,
        gen_vid=False, show_progress=False, **kwargs,
    )


def test_rollout_policy_outputs(np_env):
    n_steps = 20
    out = _rollout(np_env, make_dummy_inference_fn(np_env, 'zero'), n_steps)
    assert len(out) == 5
    frames, traj, reward_plotter, data_plotter, info_plotter = out

    assert frames is None
    assert len(traj) == n_steps + 1  # initial state + one per step
    assert isinstance(traj[0], mm.EnvState)
    assert traj[-1].data.time == pytest.approx(n_steps * np_env.dt)

    assert isinstance(reward_plotter, plotting.RewardPlotter)
    assert len(reward_plotter.rewards) == n_steps
    assert all(len(v) == n_steps for v in reward_plotter.axdata.values())

    assert isinstance(data_plotter, plotting.MujocoPlotter)
    assert isinstance(info_plotter, plotting.InfoPlotter)
    assert len(info_plotter.data['time']) == n_steps


def test_rollout_policy_info_hooks(np_env):
    out = _rollout(
        np_env, make_dummy_inference_fn(np_env, 'zero'), 5,
        info_init_fn=lambda s: {'counter': np.array(0)},
        info_step_fn=lambda s: {'counter': s.info['counter'] + 1},
        info_plot_key=['counter'],
    )
    traj, info_plotter = out[1], out[4]
    assert traj[-1].info['counter'] == len(traj) - 1
    np.testing.assert_array_equal(info_plotter.data['counter'], np.arange(1, len(traj)))


def test_rollout_policy_stops_on_done(np_env):
    # flip the cheetah over; fall_termination triggers once |pitch| > 50 deg
    def flip(state):
        state.data.qpos[2] = np.deg2rad(90)
        return state.info
    out = _rollout(np_env, make_dummy_inference_fn(np_env, 'zero'), 100, info_init_fn=flip)
    traj = out[1]
    assert len(traj) == 2
    assert traj[-1].done


@pytest.mark.slow
def test_rollout_policy_renders(np_env):
    frames, *_ = mm.eval.rollout_policy(
        inference_fn=make_dummy_inference_fn(np_env, 'zero'), env=np_env, n_steps=5,
        width=64, height=48, show_progress=False,
    )
    assert len(frames) == 6
    assert frames[0].shape == (48, 64, 3)
