"""Inference utilities for loading and interacting with trained RL models."""
from pathlib import Path
from etils import epath
from glob import glob
from brax.training.agents.ppo import checkpoint
import jax
from typing import Any, Callable, Optional, List

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

def get_all_models(run_dir: Path | str, sort: bool = True) -> List[Path]:
    """Return all checkpoint directories in ``run_dir``, sorted by step if requested."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {run_dir}")
    model_files = [
        path for path in map(Path, glob(str(run_dir / '*')))
        if path.is_dir() and path.name.isdigit()
    ]
    if sort:
        model_files.sort(key=lambda x: int(x.name))
    return model_files

def get_last_model(run_dir: Path | str) -> Path:
    """Return the most recent checkpoint directory in ``run_dir``."""
    model_files = get_all_models(run_dir, sort=True)
    if not model_files:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    return model_files[-1]

def load_policy(
    run_dir: Path | str,
    deterministic: bool = True,
    checkpoint_path: Optional[Path] = None
) -> Callable:
    """Load and return a policy from ``checkpoint_path``, or the last checkpoint in
    ``run_dir``."""
    if checkpoint_path is None:
        path = get_last_model(run_dir)
    else:
        path = Path(checkpoint_path)
    print(f'Loading model at {path.as_posix()}')
    policy = checkpoint.load_policy(
        path.resolve(),
        deterministic=deterministic
    )
    return policy

def get_params(
    run_dir: Path | str,
    path: Optional[Path] = None,
    silent: bool = False
) -> Any:
    """Load and return model parameters from ``path``, or the last checkpoint in
    ``run_dir``."""
    if path is None:
        path = get_last_model(run_dir)
    path = Path(path)
    if not silent:
        print(f'Loading model at {path.as_posix()}')
    params = checkpoint.load(epath.Path(path.resolve()))
    return params
