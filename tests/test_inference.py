import jax
import pytest

from minimal_mjx.learning import inference


@pytest.fixture
def checkpoint_dir(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    for step in ['000000000010', '000000000002', '000000000100']:
        (run_dir / step).mkdir()
    (run_dir / 'config.yaml').write_text('name: run\n')
    (run_dir / 'progress.csv').write_text('')
    return run_dir


def test_get_all_models_sorted_numerically(checkpoint_dir):
    models = inference.get_all_models(checkpoint_dir)
    assert [m.name for m in models] == ['000000000002', '000000000010', '000000000100']


def test_get_all_models_skips_files(checkpoint_dir):
    names = {m.name for m in inference.get_all_models(checkpoint_dir, sort=False)}
    assert 'config.yaml' not in names and 'progress.csv' not in names


def test_get_all_models_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        inference.get_all_models(tmp_path / 'nope')


def test_get_last_model(checkpoint_dir):
    assert inference.get_last_model(checkpoint_dir) == checkpoint_dir / '000000000100'


def test_get_all_models_accepts_str(checkpoint_dir):
    assert len(inference.get_all_models(str(checkpoint_dir))) == 3


def test_get_last_model_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match='No checkpoints'):
        inference.get_last_model(tmp_path)


def test_get_step_reset_np(np_env):
    step, reset = inference.get_step_reset(np_env)
    assert step == np_env.step and reset == np_env.reset


def test_get_step_reset_jnp(jnp_env):
    step, reset = inference.get_step_reset(jnp_env)
    assert hasattr(step, 'lower') and hasattr(reset, 'lower')  # jitted
