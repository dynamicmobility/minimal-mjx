# Basic imports
import os
import dataclasses
import functools
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

# RL imports
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo.train import train as train_ppo
from brax.training.agents.ppo import networks as ppo_networks

# jax and MJX imports
from mujoco_playground import wrapper
import minimal_mjx as mm
import jax


@dataclasses.dataclass
class Resume:
    """What a continued run inherits from the checkpoints already in its directory."""

    path  : Path  # newest checkpoint, i.e. the params to restart from
    step  : int   # env steps already trained
    epoch : int   # epochs already run, one checkpoint each after the one at step 0


def setup_ppo(ppo_params: dict, network_params: dict, resumed: Resume | None = None):
    """Default handler: brax PPO. Returns ``(train_fn, network_factory)``.

    ``train_fn`` is a :func:`functools.partial` of the brax PPO trainer with all
    algorithm params pre-bound; ``network_factory`` builds the PPO networks.
    """
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        mean_kernel_init_fn=jax.nn.initializers.lecun_uniform,
        **network_params
    )
    train_fn = functools.partial(
        train_ppo, **ppo_params,
        network_factory=network_factory,
        # brax restores the checkpointed params and normalizer, not its optimizer.
        restore_checkpoint_path=None if resumed is None else str(resumed.path),
    )
    return train_fn, network_factory

_ALGO_HANDLERS = {
    'ppo': setup_ppo,
}


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


def apply_resume(ppo_params: dict, resumed: Resume) -> dict:
    """A copy of ``ppo_params`` asking the algorithm for only the steps the run still owes.

    ``num_timesteps`` is the run's whole budget, so continuing a run is a matter of
    raising it: the algorithm is handed the difference.
    """
    total = ppo_params['num_timesteps']
    if resumed.step >= total:
        raise ValueError(
            f"{resumed.path.parent} has already trained {resumed.step} of the "
            f"{total} steps in num_timesteps; raise num_timesteps to continue it"
        )
    print(
        f'Resuming {resumed.path.parent} from {resumed.step} steps (epoch '
        f'{resumed.epoch}); {total - resumed.step} steps left of {total}'
    )
    return {**ppo_params, 'num_timesteps': total - resumed.step}


def create_training_directory(run_dir, overwrite=False, resume=False) -> Path:
    """Create the run output directory, refusing to clobber an earlier run.

    If the directory already exists but holds nothing a run produced -- only a stale
    config, or the run id of a job that died before its first epoch -- it is reused.
    ``overwrite`` reuses it whatever it holds (e.g. for scratch runs), and so does
    ``resume``, for a run that continues the checkpoints already there.
    """
    # Written before training starts, so their presence alone is not training output.
    stale = {'config.yaml', mm.utils.logging.RUN_ID_FNAME}
    run_dir = Path(run_dir)
    if run_dir.exists():
        if not (resume or overwrite) and not all(
            path.is_file() and path.name in stale for path in run_dir.iterdir()
        ):
            raise FileExistsError(f"Training directory already exists: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train(
    env,
    eval_env,
    run_dir,
    ppo_params,
    network_params,
    handle_params,
    run=None,
    progress_fn=None,
    resume=False,
    overwrite=False,
):
    """Train a policy on the given environment.

    Creates ``run_dir``, resolves the training algorithm via ``handle_params``
    then wraps the resulting ``train_fn`` with progress/checkpoint callbacks and runs
    it. Saving the run's config is up to the caller (see
    :meth:`minimal_mjx.utils.config.Config.save`).

    Args:
        env: Training environment.
        eval_env: Evaluation environment used for periodic rollouts.
        run_dir: Directory for checkpoints and progress output.
        ppo_params: Brax PPO keyword arguments. ``num_timesteps`` and
            ``normalize_observations`` are also read here.
        network_params: Keyword arguments for the network factory.
        algorithm: (optional) Name of a handler in ``_ALGO_HANDLERS``. Ignored when
            ``handle_params`` is given. Defaults to ``'ppo'``.
        run: (optional) Experiment-tracking handle (e.g. a wandb run).
        handle_params: (optional) Callable
            ``(ppo_params, network_params, resumed) -> (train_fn, network_factory)``,
            where ``resumed`` is a :class:`Resume` or None.
        progress_fn: (optional) Progress callback. Defaults to
            ``mm.utils.plotting.plot_progress``.
        resume: (optional) If True, continue the checkpoints already in ``run_dir``
            instead of refusing to overwrite it. ``num_timesteps`` is then the run's
            total budget and the algorithm trains what is left of it, checkpointing
            and plotting on the step axis the earlier job stopped on. An algorithm opts
            in by reading ``resumed``; without that it retrains its whole share from
            scratch into the same directory.
        overwrite: (optional) If True, train into ``run_dir`` even if it holds an
            earlier run's output.

    Returns:
        Tuple ``(make_inference_fn, trained_params, metrics)``.
    """
    if progress_fn is None:
        progress_fn = mm.utils.plotting.plot_progress
    output_dir = create_training_directory(run_dir, overwrite=overwrite, resume=resume)
    mm.utils.logging.save_run_id(output_dir, run)

    ppo_params = dict(ppo_params)
    total_timesteps = ppo_params['num_timesteps']
    resumed = find_resume(output_dir) if resume else None
    if resume and resumed is None:
        print(f'Nothing to resume in {output_dir}; training from scratch.')
    if resumed is not None:
        ppo_params = apply_resume(ppo_params, resumed)
    # Steps the algorithm reports are its own; the run's axis carries on from here.
    step_offset = 0 if resumed is None else resumed.step

    # Resolve the training algorithm.
    train_fn, network_factory = handle_params(ppo_params, network_params, resumed)

    # PPO params are still read here for the network config and progress plot. The algo
    # was handed the steps still owed, but the plot reports against the whole budget.
    ppo_params = {**ppo_params, 'num_timesteps': total_timesteps}
    network_config = checkpoint.network_config(
        observation_size=eval_env.observation_size,
        action_size=eval_env.action_size,
        normalize_observations=ppo_params['normalize_observations'],
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
