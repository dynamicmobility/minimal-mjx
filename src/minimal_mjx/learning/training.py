# Basic imports
import os
import dataclasses
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
    resume          = learning_config.get('resume')

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        mean_kernel_init_fn=jax.nn.initializers.lecun_uniform,
        **network_params
    )
    train_fn = functools.partial(
        train_ppo, **dict(ppo_params),
        network_factory=network_factory,
        # brax restores the checkpointed params and normalizer, not its optimizer.
        restore_checkpoint_path=None if resume is None else resume['checkpoint'],
    )
    return train_fn, network_factory


# Registry mapping ``config.algorithm`` -> parameter handler. Add new training
# algorithms here. ``handle_params`` defaults to the registered handler.
_ALGO_HANDLERS = {
    'ppo': setup_ppo,
}


@dataclasses.dataclass
class Resume:
    """What a continued run inherits from the checkpoints already in its directory."""

    path  : Path  # newest checkpoint, i.e. the params to restart from
    step  : int   # env steps already trained
    epoch : int   # epochs already run, one checkpoint each after the one at step 0


def find_checkpoints(output_dir) -> list[Path]:
    """The run's checkpoint directories, named by step and ordered by it."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    return sorted(
        (p for p in output_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )


def find_resume(output_dir) -> Resume | None:
    """Where training left off in ``output_dir``, or None if it never checkpointed.

    The epoch count is the number of checkpoints past the one written at step 0, which is
    what a run writes per epoch; taking it from the directory keeps it right across
    several continuations, and across a schedule whose epochs changed size.
    """
    checkpoints = find_checkpoints(output_dir)
    if not checkpoints:
        return None
    return Resume(checkpoints[-1], int(checkpoints[-1].name), len(checkpoints) - 1)


def apply_resume(config, resumed: Resume):
    """A copy of ``config`` asking the algorithm for only the steps the run still owes.

    ``num_timesteps`` is the run's whole budget, so continuing a run is a matter of
    raising it: the algorithm is handed the difference, and ``learning_params.resume``
    tells it which state to restart from.
    """
    total = config['learning_params']['ppo_params']['num_timesteps']
    if resumed.step >= total:
        raise ValueError(
            f"{resumed.path.parent} has already trained {resumed.step} of the "
            f"{total} steps in num_timesteps; raise num_timesteps to continue it"
        )
    config = mm.utils.config.deepcopy_config(config)
    config.learning_params.ppo_params.num_timesteps = total - resumed.step
    config.learning_params.resume = mm.utils.config.create_config_dict({
        'run_dir'    : str(resumed.path.parent),
        'checkpoint' : str(resumed.path),
        'step'       : resumed.step,
        'epoch'      : resumed.epoch,
    })
    print(
        f'Resuming {resumed.path.parent} from {resumed.step} steps (epoch '
        f'{resumed.epoch}); {total - resumed.step} steps left of {total}'
    )
    return config


def create_training_directory(config, warn_github_changes=True, resume=False):
    """Create the run output directory and save the resolved config alongside it.

    If the directory already exists but holds nothing a run produced -- only a stale
    config, or the run id of a job that died before its first epoch -- reuse it and
    overwrite those in place. ``resume`` reuses it whatever it holds, for a run that
    continues the checkpoints already there.
    """
    # Written before training starts, so their presence alone is not training output.
    stale = {'config.yaml', mm.utils.logging.RUN_ID_FNAME}
    output_dir = Path(config['save_dir']) / config['name']
    if output_dir.exists():
        contents = list(output_dir.iterdir())
        if not resume and config['name'] != 'test' and not all(
            path.is_file() and path.name in stale for path in contents
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
    resume=False,
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
        resume: (optional) If True, continue the checkpoints already in the run
            directory instead of refusing to overwrite it. ``num_timesteps`` is then the
            run's total budget and the algorithm trains what is left of it, checkpointing
            and plotting on the step axis the earlier job stopped on. An algorithm opts
            in by reading ``learning_params.resume``; without that it retrains its whole
            share from scratch into the same directory.

    Returns:
        Tuple ``(make_inference_fn, trained_params, metrics)``.
    """
    if progress_fn is None:
        progress_fn = mm.utils.plotting.plot_progress
    config = mm.utils.config.create_config_dict(config)
    output_dir = create_training_directory(
        config, warn_github_changes=warn_github_changes, resume=resume
    )
    mm.utils.logging.save_run_id(output_dir, run)

    total_timesteps = config['learning_params']['ppo_params']['num_timesteps']
    resumed = find_resume(output_dir) if resume else None
    if resume and resumed is None:
        print(f'Nothing to resume in {output_dir}; training from scratch.')
    if resumed is not None:
        config = apply_resume(config, resumed)
    # Steps the algorithm reports are its own; the run's axis carries on from here.
    step_offset = 0 if resumed is None else resumed.step

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

    # PPO params are still read here for the network config and progress plot. The algo
    # was handed the steps still owed, but the plot reports against the whole budget.
    ppo_params = config_dict.ConfigDict(config['learning_params']['ppo_params'])
    ppo_params.num_timesteps = total_timesteps
    network_config = checkpoint.network_config(
        observation_size=eval_env.observation_size,
        action_size=eval_env.action_size,
        normalize_observations=ppo_params.normalize_observations,
        network_factory=network_factory,
    )

    def save_model_fn(current_step, make_policy, params, training_state=None):
        """Checkpoint the policy, and the learner state a continued run would need."""
        step = step_offset + current_step
        mm.utils.logging.save_model(
            step, make_policy, params,
            network_config = network_config,
            output_dir     = output_dir,
            run            = run,
        )
        if training_state is not None:
            mm.utils.logging.save_training_state(output_dir, step, training_state)

    x_data, y_data, y_dataerr, times = [], [], [], []
    if resumed is not None:
        mm.utils.plotting.load_progress(
            output_dir, times, x_data, y_data, y_dataerr, before=resumed.step
        )
    first = len(times)  # this job's own rows, for the timings printed below
    train_fn = functools.partial(
        train_fn,
        progress_fn=lambda num_steps, metrics: progress_fn(
            num_steps  = step_offset + num_steps,
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
    print(f"time to jit: {times[first + 1] - times[first]}")
    print(f"time to train: {times[-1] - times[first + 1]}")

    return make_inference_fn, trained_params, metrics
