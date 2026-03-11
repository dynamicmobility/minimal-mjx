# Basic imports
from pathlib import Path
from etils import epath
import numpy as np
from glob import glob
from tqdm import tqdm

# Internal imports
import minimal_mjx.utils.plotting as plotting
from minimal_mjx.utils.state import MujocoState

# RL imports
from brax.training.agents.ppo import checkpoint

# jax and MJX imports
from mujoco_playground._src.mjx_env import MjxEnv
import jax

def get_step_reset(env):
    """Returns the reset and step functions based on the backend."""
    if env._np == jax.numpy:
        print('jitting')
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
    else:
        reset = env.reset
        step = env.step
    return step, reset

def get_all_models(config: dict, sort=True) -> Path:
    """Returns the last model file in the model directory."""
    model_dir = Path(config['save_dir']) / config['name']
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    dir_files = glob(str(model_dir / '*'))
    model_files = [Path(f) for f in dir_files if '.' not in f]
    if sort:
        model_files.sort(key=lambda x: int(x.name))
    return model_files

def get_last_model(config: dict) -> Path:
    """Returns the last model file in the model directory."""
    model_files = get_all_models(config, sort=True)
    
    return model_files[-1]

def load_policy(config, deterministic=True):
    path = get_last_model(config)
    print(f'Loading model at {path.as_posix()}')
    policy = checkpoint.load_policy(
        path.resolve(),
        deterministic=deterministic
    )
    return policy

def get_params(
    config: dict,
    path: Path = None,
    silent: bool = False
):
    if path is None:
        path = get_last_model(config)
    if not silent:
        print(f'Loading model at {path.as_posix()}')
    fullpath = path.resolve()
    
    fullpath = epath.Path(fullpath)
    params = checkpoint.load(fullpath)
    return params
    
def load_policy(config, deterministic=True):
    path = get_last_model(config)
    print(f'Loading model at {path.as_posix()}')
    policy = checkpoint.load_policy(
        path.resolve(),
        deterministic=deterministic
    )
    return policy