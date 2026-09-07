import wandb
from brax.training import checkpoint
from pathlib import Path
from typing import Any
from minimal_mjx.utils.config import create_config_dict, read_config, save_config
from brax.training.agents.ppo.checkpoint import _CONFIG_FNAME
from ml_collections.config_dict import ConfigDict
from orbax import checkpoint as ocp

# Names, inside a run directory, of the state a continued run reads back.
TRAIN_STATE_DIRNAME = 'training_state'
RUN_ID_FNAME = 'wandb_run_id.txt'


def flatten_config(config, parent_key='', sep='/'):
    """Flatten a nested (dict-like) config so every leaf is keyed by its path, joined by `sep`.
    """
    items = {}
    for key, value in config.items():
        key = str(key)
        if sep in key:
            raise ValueError(
                f"Config key '{key}'{f' (under {parent_key})' if parent_key else ''} contains "
                f"'{sep}', which is reserved as the delimiter between nesting levels. Rename "
                f"the key (e.g. '{key.replace(sep, '_')}') or flatten with a different `sep`."
            )
        new_key = f'{parent_key}{sep}{key}' if parent_key else key
        if hasattr(value, 'items'):
            items.update(flatten_config(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


def unflatten_config(flat_config, sep='/'):
    """Inverse of `flatten_config`: rebuild a nested config from its `a/b/c` keys.
    """
    config = {}
    for key, value in flat_config.items():
        if key.startswith('_'):
            continue
        *parents, leaf = key.split(sep)
        node = config
        for depth, parent in enumerate(parents):
            node = node.setdefault(parent, {})
            if not isinstance(node, dict):
                raise ValueError(
                    f"Cannot unflatten '{key}': '{sep.join(parents[:depth + 1])}' is a "
                    f"value ({node!r}), not a section."
                )
        node[leaf] = value
    return config

def initialize_wandb(entity='njanwani-gatech', project='prefMORL', name='test', config={}, **kwargs):
    """Initialize and return a new W&B run."""
    return wandb.init(
        entity    = entity,
        project   = project,
        name      = name,
        config    = flatten_config(config),
        **kwargs
    )

def save_model(current_step, make_policy, params, network_config, output_dir: Path, run: wandb.Run = None):
    """Save a Brax checkpoint and optionally log it as a W&B artifact."""
    checkpoint.save(
        path=output_dir.resolve(),
        step=current_step,
        params=params,
        config=network_config,
        config_fname=_CONFIG_FNAME,
    )
    if run:
        artifact = wandb.Artifact(name=f'{run.id}_hypernetworks', type='model')
        artifact.add_dir((output_dir / f'{current_step:012d}').resolve())
        artifact.metadata['iteration'] = current_step
        run.log_artifact(artifact)

def save_training_state(output_dir: Path, step: int, state: Any) -> None:
    """Overwrite the run's learner state, so a continued run picks up mid-stream.
    # TODO: clean up this docstring
    A brax checkpoint holds what inference needs -- the policy's params and the
    observation normalizer. Training also depends on the optimizer's moments and on
    networks the policy never runs, such as a value function, so those are written here
    instead. Only the newest state is kept, since a run continues where it stopped.
    """
    ocp.PyTreeCheckpointer().save(
        (Path(output_dir) / TRAIN_STATE_DIRNAME).resolve(), (step, state), force=True
    )


def load_training_state(output_dir: Path, like: Any) -> tuple[int, Any] | None:
    """`(step, state)` restored into `like`'s pytree, or None if the run saved none.

    `like` is a freshly built state: orbax fills its arrays, so the restored state has
    the same tree structure, shapes and dtypes by construction.
    """
    path = (Path(output_dir) / TRAIN_STATE_DIRNAME).resolve()
    if not path.exists():
        return None
    return ocp.PyTreeCheckpointer().restore(path, item=(0, like))


def save_run_id(output_dir: Path, run: wandb.Run | None) -> None:
    """Record which W&B run wrote `output_dir`, for a later job to continue it."""
    if run is not None:
        (Path(output_dir) / RUN_ID_FNAME).write_text(str(run.id))


def load_run_id(output_dir: Path) -> str | None:
    """The W&B run id a previous job left in `output_dir`, or None."""
    path = Path(output_dir) / RUN_ID_FNAME
    return path.read_text().strip() if path.exists() else None


def get_latest_artifact(run: wandb.apis.public.Run, prefix: str) -> wandb.Artifact:
    """Return the last-logged artifact of `run` whose name contains `prefix`, or raise ValueError."""
    matches = [artifact for artifact in run.logged_artifacts() if prefix in artifact.name]
    if not matches:
        raise ValueError(
            f"No '{prefix}' artifact found for run {run.id}. "
            f"Logged artifacts: {[a.name for a in run.logged_artifacts()] or 'none'}"
        )
    return matches[-1]


def download_model(run_id: str, save_dir: Path | str, model_name: str,
                   entity: str, project: str, prefix: str) -> str:
    """Download config and policy checkpoint artifacts for a W&B run.
    Model name takes contains `prefix`, or raise ValueError."""
    output_dir = Path(save_dir)
    api = wandb.Api()
    run = api.run(f'{entity}/{project}/{run_id}')

    try:
        config_artifact = get_latest_artifact(run, 'config')
    except ValueError:
        config: ConfigDict = create_config_dict(unflatten_config(run.config))
    else:
        artifact_dir = config_artifact.download(root=output_dir / model_name)
        config: ConfigDict = read_config(Path(artifact_dir) / 'config.yaml')
    config['save_dir'] = str(output_dir)
    config['name'] = str(model_name)
    (output_dir / model_name).mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / model_name / 'config.yaml')

    policy_artifact = get_latest_artifact(run, prefix)
    artifact_dir = policy_artifact.download(
        root=str(output_dir / model_name / str(policy_artifact.metadata['iteration']))
    )
    return artifact_dir