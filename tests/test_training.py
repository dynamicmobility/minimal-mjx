import functools
from pathlib import Path

import pytest

import minimal_mjx as mm
from minimal_mjx.learning import training


@pytest.fixture
def cfg(cheetah_yaml):
    return mm.utils.Config(cheetah_yaml)


# ---------------------------------------------------------------------------
# create_training_directory
# ---------------------------------------------------------------------------

def test_create_training_directory_creates_nested(tmp_path):
    run_dir = tmp_path / 'a' / 'b' / 'run'
    assert training.create_training_directory(run_dir) == run_dir
    assert run_dir.is_dir()
    assert list(run_dir.iterdir()) == []  # no config is written


def test_create_training_directory_refuses_existing_run(tmp_path):
    (tmp_path / '000000000000').mkdir()
    with pytest.raises(FileExistsError):
        training.create_training_directory(tmp_path)


@pytest.mark.parametrize('flag', ['overwrite', 'resume'])
def test_create_training_directory_reuses_existing_run(tmp_path, flag):
    (tmp_path / '000000000000').mkdir()
    assert training.create_training_directory(tmp_path, **{flag: True}) == tmp_path


def test_create_training_directory_reuses_stale_dir(tmp_path):
    (tmp_path / 'config.yaml').write_text('name: x\n')
    (tmp_path / mm.utils.logging.RUN_ID_FNAME).write_text('abc')
    assert training.create_training_directory(tmp_path) == tmp_path


def test_config_saved_into_training_directory(cfg, monkeypatch):
    """The flow example/train.py uses: create the dir, then save the config into it."""
    monkeypatch.setattr(mm.utils.config, 'get_commit_hash', lambda warn=True: 'abc123')
    cfg.name = 'my_run'
    run_dir = training.create_training_directory(cfg.run_dir, overwrite=cfg.is_test)
    cfg.save(run_dir / 'config.yaml', make_dir=False)
    assert mm.utils.read_config(run_dir / 'config.yaml').git_hash == 'abc123'
    # a stale config alone does not block a fresh start
    training.create_training_directory(cfg.run_dir, overwrite=cfg.is_test)


# ---------------------------------------------------------------------------
# resume helpers
# ---------------------------------------------------------------------------

def test_find_resume(tmp_path):
    assert training.find_resume(tmp_path) is None
    for step in [0, 100, 50]:
        (tmp_path / f'{step:012d}').mkdir()
    resumed = training.find_resume(tmp_path)
    assert resumed.step == 100
    assert resumed.epoch == 2
    assert resumed.path == tmp_path / f'{100:012d}'


def test_apply_resume_returns_remaining_steps():
    ppo_params = {'num_timesteps': 1000, 'num_envs': 8}
    resumed = training.Resume(Path('run/000000000400'), 400, 2)
    out = training.apply_resume(ppo_params, resumed)
    assert out == {'num_timesteps': 600, 'num_envs': 8}
    assert ppo_params['num_timesteps'] == 1000  # not mutated


def test_apply_resume_finished_run_raises():
    resumed = training.Resume(Path('run/000000001000'), 1000, 5)
    with pytest.raises(ValueError, match='raise num_timesteps'):
        training.apply_resume({'num_timesteps': 1000}, resumed)


# ---------------------------------------------------------------------------
# setup_ppo / train
# ---------------------------------------------------------------------------

def test_setup_ppo_binds_params(cfg):
    train_fn, network_factory = training.setup_ppo(cfg.ppo_params, cfg.network_params)
    assert isinstance(train_fn, functools.partial)
    for k, v in cfg.ppo_params.items():
        assert train_fn.keywords[k] == v
    assert train_fn.keywords['network_factory'] is network_factory
    assert train_fn.keywords['restore_checkpoint_path'] is None
    for k, v in cfg.network_params.items():
        assert network_factory.keywords[k] == v


def test_setup_ppo_restores_from_resume(cfg):
    resumed = training.Resume(Path('run/000000000400'), 400, 2)
    train_fn, _ = training.setup_ppo(cfg.ppo_params, cfg.network_params, resumed)
    assert train_fn.keywords['restore_checkpoint_path'] == 'run/000000000400'


def _train(cfg, env, **kwargs):
    return training.train(
        env, env,
        run_dir=cfg.run_dir,
        train_fn_params=cfg.ppo_params,
        network_factory_params=cfg.network_params,
        **{'handle_params': training.setup_ppo, **kwargs},
    )


def test_train_passes_params_to_handle_params(cfg, np_env):
    class Called(Exception):
        pass

    def handle_params(train_fn_params, network_factory_params, resumed):
        raise Called(train_fn_params, network_factory_params, resumed)

    with pytest.raises(Called) as exc_info:
        _train(cfg, np_env, handle_params=handle_params)
    assert exc_info.value.args == (cfg.ppo_params, cfg.network_params, None)


def test_train_resume_hands_handler_remaining_steps(cfg, np_env):
    (cfg.run_dir / f'{400:012d}').mkdir(parents=True)
    seen = {}

    def handle_params(ppo_params, network_params, resumed):
        seen.update(ppo_params=ppo_params, resumed=resumed)
        raise RuntimeError('stop')

    with pytest.raises(RuntimeError, match='stop'):
        _train(cfg, np_env, handle_params=handle_params, resume=True)
    assert seen['resumed'].step == 400
    assert seen['ppo_params']['num_timesteps'] == cfg.ppo_params['num_timesteps'] - 400


def test_train_refuses_existing_run(cfg, np_env):
    (cfg.run_dir / f'{0:012d}').mkdir(parents=True)
    with pytest.raises(FileExistsError):
        _train(cfg, np_env)


@pytest.mark.slow
def test_train_smoke_and_load_policy(cfg, jnp_env):
    ppo_params = {
        **cfg.ppo_params,
        'num_timesteps': 2000, 'num_envs': 8, 'batch_size': 8, 'num_minibatches': 1,
        'unroll_length': 5, 'episode_length': 50, 'num_evals': 2, 'num_updates_per_batch': 1,
    }
    network_params = {'policy_hidden_layer_sizes': [8], 'value_hidden_layer_sizes': [8]}
    _, params, _ = training.train(
        jnp_env, jnp_env, cfg.run_dir, ppo_params, network_params,
        handle_params=training.setup_ppo, overwrite=True,
    )
    assert params is not None

    assert mm.learning.inference.get_last_model(cfg.run_dir).is_dir()
    policy = mm.learning.inference.load_policy(cfg.run_dir)
    import jax
    state = jax.jit(jnp_env.reset)(jax.random.PRNGKey(0))
    action, _ = policy(state.obs, jax.random.PRNGKey(0))
    assert action.shape == (jnp_env.action_size,)
