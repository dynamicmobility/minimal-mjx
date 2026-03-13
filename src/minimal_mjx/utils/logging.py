import wandb
from brax.training import checkpoint
from pathlib import Path
from minimal_mjx.utils.config import read_config, save_config, get_commit_hash
from brax.training.agents.ppo.checkpoint import _CONFIG_FNAME
from ml_collections.config_dict import ConfigDict
import os


def initialize_wandb(entity='njanwani-gatech', project='prefMORL', name='test', config={}, **kwargs):
    """Initialize and return a new W&B run."""
    return wandb.init(entity=entity, project=project, name=name, config=config, **kwargs)

def begin_training_log(config):
    # Make training directory
    output_dir = Path(config['save_dir']) / config['name']
    os.makedirs(output_dir, exist_ok=config['name'] == 'test')

    # Save config in directory
    config_save_path = Path(output_dir) / 'config.yaml'
    if config.name != 'test':
        git_hash = get_commit_hash()
        config.git_hash = git_hash
    save_config(config, config_save_path)


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

def _find_artifact(run: wandb.apis.public.Run, prefix: str) -> wandb.Artifact:
    """Return the latest artifact whose name contains `prefix`, or raise ValueError."""
    match = None
    for artifact in run.logged_artifacts():
        if prefix in artifact.name:
            match = artifact
    if match is None:
        raise ValueError(f"No '{prefix}' artifact found for run {run.id}")
    return match


def download_model(run_id: str, save_dir: Path | str, model_name: str,
                   entity: str = 'njanwani-gatech', project: str = 'prefMORL') -> str:
    """Download config and policy checkpoint artifacts for a W&B run."""
    output_dir = Path(save_dir)
    api = wandb.Api()
    run = api.run(f'{entity}/{project}/{run_id}')

    config_artifact = _find_artifact(run, 'config')
    artifact_dir = config_artifact.download(root=output_dir / model_name)
    config: ConfigDict = read_config(Path(artifact_dir) / 'config.yaml')
    config['save_dir'] = str(output_dir)
    config['name'] = str(model_name)
    save_config(config, output_dir / model_name / 'config.yaml')

    policy_artifact = _find_artifact(run, 'hypernetworks')
    artifact_dir = policy_artifact.download(
        root=str(output_dir / model_name / str(policy_artifact.metadata['iteration']))
    )
    return artifact_dir