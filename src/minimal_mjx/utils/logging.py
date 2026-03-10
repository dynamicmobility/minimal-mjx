import wandb

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
        artifact.add_dir(output_dir.resolve())
        run.log_artifact(artifact)
