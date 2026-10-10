import jax
import pytest

from minimal_mjx.learning import inference


@pytest.fixture
def run_dir_config(tmp_path):
    config = {'save_dir': str(tmp_path), 'name': 'run'}
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    for step in ['000000000010', '000000000002', '000000000100']:
        (run_dir / step).mkdir()
    (run_dir / 'config.yaml').write_text('name: run\n')
    (run_dir / 'progress.csv').write_text('')
    return config, run_dir


def test_get_all_models_sorted_numerically(run_dir_config):
    config, run_dir = run_dir_config
    models = inference.get_all_models(config)
    assert [m.name for m in models] == ['000000000002', '000000000010', '000000000100']


def test_get_all_models_skips_files(run_dir_config):
    config, _ = run_dir_config
    names = {m.name for m in inference.get_all_models(config, sort=False)}
    assert 'config.yaml' not in names and 'progress.csv' not in names


def test_get_all_models_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        inference.get_all_models({'save_dir': str(tmp_path), 'name': 'nope'})


def test_get_last_model(run_dir_config):
    config, run_dir = run_dir_config
    assert inference.get_last_model(config) == run_dir / '000000000100'


def test_get_step_reset_np(np_env):
    step, reset = inference.get_step_reset(np_env)
    assert step == np_env.step and reset == np_env.reset


def test_get_step_reset_jnp(jnp_env):
    step, reset = inference.get_step_reset(jnp_env)
    assert hasattr(step, 'lower') and hasattr(reset, 'lower')  # jitted
