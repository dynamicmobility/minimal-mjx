import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'egl')

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CHEETAH_YAML = REPO_ROOT / 'example' / 'cheetah.yaml'

# make `example` importable
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def cheetah_dict(tmp_path):
    """Raw dict of example/cheetah.yaml with save_dir moved under tmp_path."""
    with open(CHEETAH_YAML) as f:
        data = yaml.safe_load(f)
    data['save_dir'] = str(tmp_path / 'results')
    return data


@pytest.fixture
def cheetah_yaml(tmp_path, cheetah_dict):
    """Path to a copy of the cheetah config whose save_dir is under tmp_path."""
    path = tmp_path / 'cheetah.yaml'
    with open(path, 'w') as f:
        yaml.dump(cheetah_dict, f)
    return path


@pytest.fixture(scope='session')
def cheetah_config():
    import minimal_mjx as mm
    return mm.utils.read_config(CHEETAH_YAML)


@pytest.fixture(scope='session')
def np_env(cheetah_config):
    from example.create_environment import create_environment
    env, _ = create_environment(cheetah_config)
    return env


@pytest.fixture(scope='session')
def jnp_env(cheetah_config):
    from example.create_environment import create_environment
    env, _ = create_environment(cheetah_config, for_training=True)
    return env


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """A fresh git repo with one commit, set as the cwd."""
    import subprocess
    repo = tmp_path / 'repo'
    repo.mkdir()
    run = lambda *args: subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)
    run('init', '-q')
    run('config', 'user.email', 'test@example.com')
    run('config', 'user.name', 'test')
    (repo / 'file.txt').write_text('hello')
    run('add', '.')
    run('commit', '-q', '-m', 'init')
    monkeypatch.chdir(repo)
    return repo
