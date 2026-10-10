import functools

import pytest
from ml_collections import config_dict

import minimal_mjx as mm
from minimal_mjx.learning import training


@pytest.fixture
def cfg(cheetah_dict):
    return config_dict.ConfigDict(cheetah_dict)


def test_create_training_directory_test_run(cfg, monkeypatch):
    monkeypatch.setattr(mm.utils.config, 'get_commit_hash', lambda **_: pytest.fail('test runs skip git'))
    out = training.create_training_directory(cfg)
    assert out.is_dir()
    saved = mm.utils.read_config(out / 'config.yaml')
    assert 'git_hash' not in saved
    training.create_training_directory(cfg)  # overwriting is allowed


def test_create_training_directory_named_run(cfg, monkeypatch):
    monkeypatch.setattr(mm.utils.config, 'get_commit_hash', lambda warn=True: 'abc123')
    cfg.name = 'my_run'
    out = training.create_training_directory(cfg)
    assert mm.utils.read_config(out / 'config.yaml').git_hash == 'abc123'
    with pytest.raises(FileExistsError):
        training.create_training_directory(cfg)


def test_setup_ppo_binds_params(cfg):
    train_fn, network_factory = training.setup_ppo(cfg)
    assert isinstance(train_fn, functools.partial)
    for k, v in cfg.learning_params.ppo_params.items():
        assert train_fn.keywords[k] == v
    assert train_fn.keywords['network_factory'] is network_factory
    for k, v in cfg.learning_params.network_params.items():
        assert network_factory.keywords[k] == v


def test_train_unknown_algorithm(cfg, np_env):
    cfg.algorithm = 'sac'
    with pytest.raises(ValueError, match="Unknown algorithm 'sac'"):
        training.train(cfg, np_env, np_env)


def test_train_uses_custom_handle_params(cfg, np_env):
    class Called(Exception):
        pass

    def handle_params(config):
        raise Called(config.name)

    with pytest.raises(Called, match='test'):
        training.train(cfg, np_env, np_env, handle_params=handle_params)


@pytest.mark.slow
def test_train_smoke_and_load_policy(cfg, jnp_env):
    ppo = cfg.learning_params.ppo_params
    ppo.update(
        num_timesteps=2000, num_envs=8, batch_size=8, num_minibatches=1,
        unroll_length=5, episode_length=50, num_evals=2, num_updates_per_batch=1,
    )
    cfg.learning_params.network_params.update(
        policy_hidden_layer_sizes=[8], value_hidden_layer_sizes=[8],
    )
    _, params, _ = training.train(cfg, jnp_env, jnp_env)
    assert params is not None

    run_dir = mm.learning.inference.get_last_model(cfg)
    assert run_dir.is_dir()
    policy = mm.learning.inference.load_policy(cfg)
    import jax, jax.numpy as jnp
    state = jax.jit(jnp_env.reset)(jax.random.PRNGKey(0))
    action, _ = policy(state.obs, jax.random.PRNGKey(0))
    assert action.shape == (jnp_env.action_size,)
