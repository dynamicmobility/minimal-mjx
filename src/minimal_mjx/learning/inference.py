"""Inference utilities for loading and interacting with trained RL models."""
from pathlib import Path
from etils import epath
from glob import glob
from brax.training.agents.ppo import checkpoint
import jax
from typing import Any, Callable, Optional, Dict, List

def get_step_reset(env: Any) -> tuple[Callable, Callable]:
    """Return (step, reset) functions, jitted if using JAX backend."""
    if env._np == jax.numpy:
        print('jitting')
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
    else:
        reset = env.reset
        step = env.step
    return step, reset

def get_all_models(config: Dict[str, Any], sort: bool = True) -> List[Path]:
    """Return all model directories as Path objects, sorted by integer name if requested."""
    model_dir = Path(config['save_dir']) / config['name']
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    dir_files = glob(str(model_dir / '*'))
    model_files = [Path(f) for f in dir_files if '.' not in f]
    if sort:
        model_files.sort(key=lambda x: int(x.name))
    return model_files

def get_last_model(config: Dict[str, Any]) -> Path:
    """Return the most recent model directory as a Path object."""
    model_files = get_all_models(config, sort=True)
    return model_files[-1]

def load_policy(config: Dict[str, Any], deterministic: bool = True) -> Callable:
    """Load and return a policy from the last model checkpoint."""
    path = get_last_model(config)
    print(f'Loading model at {path.as_posix()}')
    policy = checkpoint.load_policy(
        path.resolve(),
        deterministic=deterministic
    )
    return policy

def get_params(
    config: Dict[str, Any],
    path: Optional[Path] = None,
    silent: bool = False
) -> Any:
    """Load and return model parameters from a checkpoint."""
    if path is None:
        path = get_last_model(config)
    if not silent:
        print(f'Loading model at {path.as_posix()}')
    fullpath = path.resolve()
    fullpath = epath.Path(fullpath)
    params = checkpoint.load(fullpath)
    return params