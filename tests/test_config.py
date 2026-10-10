from pathlib import Path

import pytest
import yaml
from ml_collections import config_dict

import minimal_mjx as mm
from minimal_mjx.utils import config as mmc


# ---------------------------------------------------------------------------
# read_yaml / read_config
# ---------------------------------------------------------------------------

def test_read_config_returns_configdict(cheetah_yaml, cheetah_dict):
    cfg = mmc.read_config(cheetah_yaml)
    assert isinstance(cfg, config_dict.ConfigDict)
    assert cfg.to_dict() == cheetah_dict


def test_read_config_requires_path():
    with pytest.raises(TypeError):
        mmc.read_config()


def test_read_config_ignores_argv(cheetah_yaml, monkeypatch, tmp_path):
    monkeypatch.setattr('sys.argv', ['script.py', str(tmp_path / 'not_this.yaml')])
    assert mmc.read_config(cheetah_yaml).env == 'Cheetah'


def test_read_yaml_missing_file_raises(tmp_path):
    missing = tmp_path / 'missing.yaml'
    with pytest.raises(Exception, match='missing.yaml') as exc_info:
        mmc.read_yaml(missing)
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_read_yaml_malformed_raises(tmp_path):
    bad = tmp_path / 'bad.yaml'
    bad.write_text('a: [1, 2\n')
    with pytest.raises(Exception, match='bad.yaml'):
        mmc.read_yaml(bad)


# ---------------------------------------------------------------------------
# create_config_dict / save_config
# ---------------------------------------------------------------------------

def test_create_config_dict_is_recursive():
    cfg = mmc.create_config_dict({'a': 1, 'b': {'c': {'d': [1, 2]}}})
    assert isinstance(cfg, config_dict.ConfigDict)
    assert isinstance(cfg.b, config_dict.ConfigDict)
    assert isinstance(cfg.b.c, config_dict.ConfigDict)
    assert cfg.b.c.d == [1, 2]


@pytest.mark.parametrize('as_configdict', [False, True])
def test_save_config_round_trip(tmp_path, cheetah_dict, as_configdict):
    cfg = config_dict.ConfigDict(cheetah_dict) if as_configdict else cheetah_dict
    path = tmp_path / 'out.yaml'
    mmc.save_config(cfg, path)
    assert mmc.read_config(path).to_dict() == cheetah_dict


def test_save_config_uses_flow_style_lists(tmp_path):
    path = tmp_path / 'out.yaml'
    mmc.save_config({'sizes': [64, 64], 'nested': {'x': [1, 2, 3]}}, path)
    text = path.read_text()
    assert 'sizes: [64, 64]' in text
    assert 'x: [1, 2, 3]' in text


# ---------------------------------------------------------------------------
# get_commit_hash
# ---------------------------------------------------------------------------

def test_get_commit_hash_clean_repo(git_repo, monkeypatch):
    import subprocess
    expected = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    monkeypatch.setattr('builtins.input', lambda *_: pytest.fail('should not prompt'))
    assert mmc.get_commit_hash(warn=True) == expected


def test_get_commit_hash_dirty_repo_prompts_only_when_warn(git_repo, monkeypatch):
    (git_repo / 'file.txt').write_text('changed')
    prompts = []
    monkeypatch.setattr('builtins.input', lambda msg='': prompts.append(msg))

    assert mmc.get_commit_hash(warn=False) is not None
    assert prompts == []

    assert mmc.get_commit_hash(warn=True) is not None
    assert len(prompts) == 1


def test_get_commit_hash_outside_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path.parent))
    assert mmc.get_commit_hash(warn=False) is None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_properties(cheetah_yaml, cheetah_dict):
    cfg = mmc.Config(cheetah_yaml)
    lp = cheetah_dict['learning_params']

    assert cfg.name == cheetah_dict['name']
    assert cfg.algorithm == cheetah_dict['algorithm']
    assert cfg.env_name == cheetah_dict['env']
    assert cfg.backend == cheetah_dict['backend']
    assert cfg.save_dir == Path(cheetah_dict['save_dir'])
    assert cfg.run_dir == Path(cheetah_dict['save_dir']) / cheetah_dict['name']
    assert cfg.ppo_params == lp['ppo_params']
    assert cfg.network_params == lp['network_params']
    assert cfg.normalize_observations == lp['ppo_params']['normalize_observations']


def test_config_param_getters_return_plain_dicts(cheetah_yaml):
    cfg = mmc.Config(cheetah_yaml)
    assert type(cfg.ppo_params) is dict
    assert type(cfg.network_params) is dict


def test_config_env_params_is_independent_copy(cheetah_yaml, cheetah_dict):
    cfg = mmc.Config(cheetah_yaml)
    env_params = cfg.env_params
    assert isinstance(env_params, config_dict.ConfigDict)
    assert isinstance(env_params.reward, config_dict.ConfigDict)
    assert env_params.to_dict() == cheetah_dict['env_config']

    env_params.ctrl_dt = 123.0
    env_params.reward.weights.run = -1.0
    assert cfg.env_params.ctrl_dt == cheetah_dict['env_config']['ctrl_dt']
    assert cfg.env_params.reward.weights.run == cheetah_dict['env_config']['reward']['weights']['run']


def test_config_load_replaces_cfg(cheetah_yaml, cheetah_dict, tmp_path):
    other = dict(cheetah_dict, name='other')
    other_path = tmp_path / 'other.yaml'
    mmc.save_config(other, other_path)

    cfg = mmc.Config(cheetah_yaml)
    returned = cfg.load(other_path)
    assert cfg.name == 'other'
    assert returned is cfg.cfg


def test_config_works_with_environment_factory(cheetah_yaml):
    """The properties are what scripts hand to environment factories."""
    from example.cheetah import Cheetah
    cfg = mmc.Config(cheetah_yaml)
    env = Cheetah(env_params=cfg.env_params, backend=cfg.backend)
    assert env.backend == cfg.backend


# ---------------------------------------------------------------------------
# Config.save
# ---------------------------------------------------------------------------

def test_config_save_test_run(cheetah_yaml, monkeypatch):
    monkeypatch.setattr(mmc, 'get_commit_hash', lambda **_: pytest.fail('test runs skip git'))
    cfg = mmc.Config(cheetah_yaml)
    assert cfg.name == 'test'

    out = cfg.run_dir / 'config.yaml'
    cfg.save(out)
    assert out.exists()
    assert mmc.Config(out).cfg.to_dict() == cfg.cfg.to_dict()
    assert 'git_hash' not in cfg.cfg

    # test runs may be overwritten
    cfg.save(out)


def test_config_save_named_run_records_git_hash(cheetah_yaml, monkeypatch):
    monkeypatch.setattr(mmc, 'get_commit_hash', lambda warn=True: 'abc123')
    cfg = mmc.Config(cheetah_yaml)
    cfg.cfg.name = 'my_run'

    out = cfg.run_dir / 'config.yaml'
    cfg.save(out)
    assert cfg.run_dir.is_dir()
    assert mmc.read_config(out).git_hash == 'abc123'


def test_config_save_named_run_refuses_existing_dir(cheetah_yaml, monkeypatch):
    monkeypatch.setattr(mmc, 'get_commit_hash', lambda warn=True: 'abc123')
    cfg = mmc.Config(cheetah_yaml)
    cfg.cfg.name = 'my_run'
    cfg.save(cfg.run_dir / 'config.yaml')
    with pytest.raises(FileExistsError):
        cfg.save(cfg.run_dir / 'config.yaml')


def test_config_save_rewrite_test_false_skips_git(cheetah_yaml, monkeypatch):
    monkeypatch.setattr(mmc, 'get_commit_hash', lambda **_: pytest.fail('should skip git'))
    cfg = mmc.Config(cheetah_yaml)
    cfg.cfg.name = 'my_run'
    cfg.save(cfg.run_dir / 'config.yaml', rewrite_test=False)
    assert 'git_hash' not in cfg.cfg


def test_config_save_without_make_dir(cheetah_yaml, tmp_path):
    cfg = mmc.Config(cheetah_yaml)
    out = tmp_path / 'flat.yaml'
    cfg.save(out, make_dir=False)
    assert out.exists()
    assert not cfg.run_dir.exists()


def test_config_save_uses_flow_style_lists(cheetah_yaml):
    cfg = mmc.Config(cheetah_yaml)
    out = cfg.run_dir / 'config.yaml'
    cfg.save(out)
    assert 'policy_hidden_layer_sizes: [64, 64]' in out.read_text()


# ---------------------------------------------------------------------------
# Subclassing (moplayground's MOConfig extends Config)
# ---------------------------------------------------------------------------

class _MOConfig(mmc.Config):

    @property
    def network_params(self) -> dict:
        return self.cfg.learning_params[f'{self.algorithm}_params'].to_dict()

    @property
    def num_objectives(self) -> int:
        return len(self.cfg.env_config.reward.optimization.objectives)


def test_subclass_overrides_and_extends(tmp_path, cheetah_dict):
    data = dict(cheetah_dict, algorithm='morlax')
    data['learning_params'] = dict(
        data['learning_params'],
        morlax_params={'hypertype': 'dual', 'policy_hidden_layer_sizes': [32]},
    )
    data['env_config']['reward']['optimization'] = {'objectives': [['run'], ['energy']]}
    path = tmp_path / 'mo.yaml'
    mmc.save_config(data, path)

    cfg = _MOConfig(path)
    assert cfg.network_params == {'hypertype': 'dual', 'policy_hidden_layer_sizes': [32]}
    assert cfg.num_objectives == 2
    assert cfg.run_dir == Path(data['save_dir']) / 'test'

    out = cfg.run_dir / 'config.yaml'
    cfg.save(out)
    assert _MOConfig(out).num_objectives == 2


def test_read_config_exported_from_utils():
    assert mm.utils.read_config is mmc.read_config


# ---------------------------------------------------------------------------
# Setters / defaults
# ---------------------------------------------------------------------------

def test_config_name_and_save_dir_setters(cheetah_yaml, tmp_path):
    cfg = mmc.Config(cheetah_yaml)
    cfg.name = 'renamed'
    cfg.save_dir = tmp_path / 'elsewhere'
    assert cfg.cfg.name == 'renamed'
    assert cfg.cfg.save_dir == str(tmp_path / 'elsewhere')  # stays YAML-serializable
    assert cfg.run_dir == tmp_path / 'elsewhere' / 'renamed'


def test_config_is_test(cheetah_yaml):
    cfg = mmc.Config(cheetah_yaml)
    assert cfg.is_test
    cfg.name = 'real'
    assert not cfg.is_test


def test_config_save_default_path(cheetah_yaml):
    cfg = mmc.Config(cheetah_yaml)
    cfg.save()
    assert (cfg.run_dir / 'config.yaml').exists()


def test_config_algorithm_defaults_to_ppo(tmp_path, cheetah_dict):
    del cheetah_dict['algorithm']
    path = tmp_path / 'noalgo.yaml'
    mmc.save_config(cheetah_dict, path)
    assert mmc.Config(path).algorithm == 'ppo'


def test_deepcopy_config_is_independent():
    cfg = config_dict.ConfigDict({'a': {'b': [1, 2]}})
    copied = mmc.deepcopy_config(cfg)
    copied.a.b.append(3)
    assert cfg.a.b == [1, 2]


def test_config_exported():
    assert mm.Config is mmc.Config
    assert mm.utils.Config is mmc.Config
