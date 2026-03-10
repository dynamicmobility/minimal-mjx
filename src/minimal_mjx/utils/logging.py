import wandb
from brax.training import checkpoint
from pathlib import Path
import json

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
        path   = output_dir.resolve(),
        step   = current_step,
        params = params,
        config = network_config
    )
    if run:
        artifact = wandb.Artifact(
            name=f"{current_step}",
            type="model"
        )
        artifact.add_dir((output_dir / current_step).resolve())
        run.log_artifact(artifact)

def log_config(run: wandb.Run, config: dict):
    artifact = wandb.Artifact(name="config", type="config")
    artifact.add_dir(config) if isinstance(config, Path) else artifact.add_metadata(config)
    run.log_artifact(artifact)
    
def download_config(run: wandb.Run, name: str = "config") -> dict:
    artifact = run.use_artifact(name)
    artifact_dir = artifact.download()
    config_path = Path(artifact_dir) / "config.yaml"
    return config_path

def download_policy(run: wandb.Run, name: str, output_dir: Path | str):
    if type(output_dir) == str:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact = run.use_artifact(name)
    artifact_dir = artifact.download(root=str(output_dir))
    return artifact_dir


if __name__ == '__main__':
    run = initialize_wandb()
    dir = download_policy(run, name='njanwani-gatech/prefMORL/180940800:latest', output_dir='results/bruce')