import wandb
from brax.training import checkpoint
from pathlib import Path
from minimal_mjx.utils.config import read_config, save_config
from brax.training.agents.ppo.checkpoint import _CONFIG_FNAME
from ml_collections.config_dict import ConfigDict

def initialize_wandb(
    entity='njanwani-gatech',
    project='prefMORL',
    name='test',
    config={},
    **kwargs
):
    # Start a new wandb run to track this script.
    run = wandb.init(
        entity    = entity,
        project   = project,
        name      = name,
        config    = config,
        **kwargs
    )
    return run

def save_model(current_step, make_policy, params, network_config, output_dir: Path, run: wandb.Run = None):
    checkpoint.save(
        path            = output_dir.resolve(),
        step            = current_step,
        params          = params,
        config          = network_config,
        config_fname    = _CONFIG_FNAME
    )
    if run:
        artifact = wandb.Artifact(
            name=f'{run.id}_hypernetworks',
            type="model"
        )
        artifact.add_dir((output_dir / f'{current_step:012d}').resolve())
        artifact.metadata['iteration'] = current_step
        run.log_artifact(artifact)

def _find_artifact(run: wandb.apis.public.Run, prefix: str) -> wandb.Artifact:
    """Find the latest logged artifact whose name starts with the given prefix.

    Iterates through all artifacts logged by the run and returns the last match,
    which corresponds to the most recent version.

    Args:
        run: A W&B run object.
        prefix: The artifact name prefix to search for.

    Returns:
        The matching wandb.Artifact.

    Raises:
        ValueError: If no artifact with the given prefix is found.
    """
    match = None
    for artifact in run.logged_artifacts():
        if prefix in artifact.name:
            match = artifact
    if match is None:
        raise ValueError(f"No '{prefix}' artifact found for run {run.id}")
    return match

def download_model(
    run_id        : str,
    save_dir      : Path | str,
    model_name    : str,
    entity        : str = 'njanwani-gatech',
    project       : str = 'prefMORL',
) -> str:
    """Download the policy model artifact for a specific W&B run.

    Retrieves both the config and the latest actor-critic hypernetwork checkpoint
    logged during the run, saving them under ``output_dir``.

    Args:
        run_id: The unique run identifier.
        save_dir: Save directory to download artifacts into.
        model_name: sub-directory within output_dir where model files download to 
        entity: W&B entity (user or team).
        project: W&B project name.

    Returns:
        The local path to the downloaded policy checkpoint directory.
    """
    output_dir    = Path(save_dir)
    api           = wandb.Api()
    run           = api.run(f'{entity}/{project}/{run_id}')
    
    config_artifact       = _find_artifact(run, 'config')
    artifact_dir          = config_artifact.download(root=output_dir / model_name)
    config: ConfigDict    = read_config(Path(artifact_dir) / 'config.yaml')
    config['save_dir']    = str(output_dir)
    config['name']        = str(model_name)
    save_config(config, output_dir / model_name / 'config.yaml')

    policy_artifact   = _find_artifact(run, 'hypernetworks')
    artifact_dir      = policy_artifact.download(
        root=str(output_dir / model_name / str(policy_artifact.metadata['iteration']))
    )
    return artifact_dir