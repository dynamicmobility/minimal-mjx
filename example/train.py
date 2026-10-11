import os
os.environ["MUJOCO_GL"] = "egl"
import minimal_mjx as mm
from pathlib import Path
from example.create_environment import create_environment
import argparse
import jax
import wandb

def main(
    config_path  : Path,
    use_wandb    : bool = True,
    wandb_entity : str = 'njanwani-gatech',
    wandb_project: str = 'my-awesome-project-2',
    T            : float = 10.0,
    camera       : str = 'track'
):
    # Read configuration file
    cfg = mm.utils.Config(config_path)

    # Create environment
    env, env_params = create_environment(cfg, for_training=True)

    # Create the run directory and save the config next to the checkpoints
    # (runs named 'test' may be overwritten and skip recording the git hash)
    run_dir = mm.create_training_directory(cfg.run_dir, overwrite=cfg.is_test)
    cfg.save(run_dir / 'config.yaml', make_dir=False)

    # Initialize Weights and Biases Run object (if you want to use it)
    if not use_wandb:
        run = None
    else:
        run = mm.initialize_wandb(
            name    = str(run_dir).replace('/', ''),
            entity  = wandb_entity,
            project = wandb_project,
            config  = cfg.cfg.to_dict()
        )

    # Begin training
    make_inference_fn, params, _ = mm.train(
        env                       = env,
        eval_env                  = env,
        run_dir                   = run_dir,
        handle_params             = mm.setup_ppo,
        train_fn_params           = cfg.ppo_params,
        network_factory_params    = cfg.network_params,
        run                       = run,
        overwrite                 = cfg.is_test,
    )

    # Save a video of the trained policy
    if T > 0:
        inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
        frames, *_ = mm.rollout_policy(
            inference_fn = inference_fn,
            env          = env,
            n_steps      = int(T / env.dt),
            camera       = camera,
            width        = 640,
            height       = 480
        )
        video_path = run_dir / 'rollout.mp4'
        mm.save_video(frames, env.dt, video_path)
        if run is not None:
            run.log({'rollout': wandb.Video(str(video_path), format='mp4')})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a policy from a config file.')
    parser.add_argument('--config', type=Path, required=True, help='Path to the config file')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights and Biases logging')
    parser.add_argument('--wandb_entity', default='njanwani-gatech', help='Weights and Biases entity')
    parser.add_argument('--wandb_project', default='my-awesome-project-2', help='Weights and Biases project')
    parser.add_argument('--T', type=float, default=10.0,
                        help='Seconds of the trained policy to save as a video (0 to skip)')
    parser.add_argument('--camera', default='track', help='Camera name to render the video from')
    args = parser.parse_args()

    main(
        config_path   = args.config,
        use_wandb     = not args.no_wandb,
        wandb_entity  = args.wandb_entity,
        wandb_project = args.wandb_project,
        T             = args.T,
        camera        = args.camera
    )
