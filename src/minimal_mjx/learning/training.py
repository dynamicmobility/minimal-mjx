# Basic imports
import os
import functools
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from ml_collections import config_dict

# RL imports
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo.train import train as train_ppo
from brax.training.agents.ppo import networks as ppo_networks

# jax and MJX imports
from mujoco_playground import wrapper
import minimal_mjx as mm
import jax


def setup_ppo(config):
    """Default handler: brax PPO. Returns ``(train_fn, network_factory)``.

    ``train_fn`` is a :func:`functools.partial` of the brax PPO trainer with all
    algorithm params pre-bound; ``network_factory`` builds the PPO networks.
    """
    learning_config = config['learning_params']
    ppo_params      = config_dict.ConfigDict(learning_config['ppo_params'])
    network_params  = config_dict.ConfigDict(learning_config['network_params'])

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        mean_kernel_init_fn=jax.nn.initializers.lecun_uniform,
        **network_params
    )
    train_fn = functools.partial(
        train_ppo, **dict(ppo_params),
        network_factory=network_factory,
    )
    return train_fn, network_factory


# Registry mapping ``config.algorithm`` -> parameter handler. Add new training
# algorithms here. ``handle_params`` defaults to the registered handler.
_ALGO_HANDLERS = {
    'ppo': setup_ppo,
}


def create_training_directory(config, warn_github_changes=True):
    """Create the run output directory and save the resolved config alongside it.

    If the directory already exists but only contains a stale config file,
    reuse it and overwrite that config in place.
    """
    output_dir = Path(config['save_dir']) / config['name']
    if output_dir.exists():
        contents = list(output_dir.iterdir())
        if config['name'] != 'test' and not (
            len(contents) == 1 and contents[0].is_file() and contents[0].name == 'config.yaml'
        ):
            raise FileExistsError(f"Training directory already exists: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    config_save_path = Path(output_dir) / 'config.yaml'
    if config.name != 'test':
        git_hash = mm.utils.config.get_commit_hash(warn=warn_github_changes)
        config.git_hash = git_hash
    mm.utils.config.save_config(config, config_save_path)

    return output_dir


def train(
    config,
    env,
    eval_env,
    run=None,
    handle_params=None,
    warn_github_changes=False,
    progress_fn=None,
):
    """Train a policy on the given environment.

    Builds the directory/config, resolves the training algorithm via
    ``handle_params`` (defaulting to the handler registered for
    ``config.algorithm`` in ``_ALGO_HANDLERS``, or ``'ppo'`` if unset), then wraps
    the resulting ``train_fn`` with progress/checkpoint callbacks and runs it.

    Args:
        config: Training config (dict or ConfigDict). Must include ``save_dir``,
            ``name``, and ``learning_params`` (with ``ppo_params`` and
            ``network_params`` for the default PPO path).
        env: Training environment.
        eval_env: Evaluation environment used for periodic rollouts.
        run: (optional) Experiment-tracking handle (e.g. a wandb run).
        handle_params: (optional) Callable ``config -> (train_fn, network_factory)``.
            Defaults to the handler registered for ``config.algorithm``.
        warn_github_changes: (optional) If True, warn about uncommitted git changes
            when creating the training directory. Defaults to False.
        progress_fn: (optional) Progress callback. Defaults to
            ``mm.utils.plotting.plot_progress``.

    Returns:
        Tuple ``(make_inference_fn, trained_params, metrics)``.
    """
    if progress_fn is None:
        progress_fn = mm.utils.plotting.plot_progress
    config = mm.utils.config.create_config_dict(config)
    output_dir = create_training_directory(config, warn_github_changes=warn_github_changes)

    # Resolve the training algorithm.
    if handle_params is None:
        print('Using default parameter handler')
        algo = config.get('algorithm', 'ppo')
        if algo not in _ALGO_HANDLERS:
            raise ValueError(
                f"Unknown algorithm '{algo}'. Expected one of {list(_ALGO_HANDLERS)}."
            )
        handle_params = _ALGO_HANDLERS[algo]
    train_fn, network_factory = handle_params(config)

    # PPO params are still read here for the network config and progress plot.
    ppo_params = config_dict.ConfigDict(config['learning_params']['ppo_params'])
    network_config = checkpoint.network_config(
        observation_size=eval_env.observation_size,
        action_size=eval_env.action_size,
        normalize_observations=ppo_params.normalize_observations,
        network_factory=network_factory,
    )
    save_model_fn = functools.partial(
        mm.utils.logging.save_model,
        output_dir     = output_dir,
        run            = run,
        network_config = network_config,
    )

    x_data, y_data, y_dataerr, times = [], [], [], []
    train_fn = functools.partial(
        train_fn,
        progress_fn=lambda num_steps, metrics: progress_fn(
            num_steps  = num_steps,
            metrics    = metrics,
            times      = times,
            x_data     = x_data,
            y_data     = y_data,
            y_dataerr  = y_dataerr,
            ppo_params = ppo_params,
            save_dir   = output_dir,
            run        = run,
        ),
        policy_params_fn=save_model_fn,
    )

    print(
        'Started training at',
        datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")
    )
    make_inference_fn, trained_params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        eval_env=eval_env,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    return make_inference_fn, trained_params, metrics
