import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import minimal_mjx as mm
from pathlib import Path
from example.create_environment import create_environment
import jax
import argparse

def main(
    config_path      : Path,
    inference_fn_type: str = 'policy',
    model_path       : Path = None,
    T                : float = 10.0,
    width            : int = None,
    height           : int = None,
    camera           : str = 'track',
    deterministic    : bool = True,
    seed             : int = None
):
    if seed is not None:
        np.random.seed(seed)

    cfg = mm.Config(config_path)
    env, env_params = create_environment(cfg)

    if inference_fn_type == 'policy':
        inference_fn = mm.learning.inference.load_policy(
            cfg.run_dir,
            deterministic   = deterministic,
            checkpoint_path = model_path
        )
        inference_fn = jax.jit(inference_fn)
    else:
        inference_fn = mm.make_dummy_inference_fn(
            env  = env,
            mode = inference_fn_type
        )

    frames, _, reward_plotter, _, _ = mm.eval.rollout_policy(
        inference_fn    = inference_fn,
        env             = env,
        n_steps         = int(T / env.dt),
        camera          = camera,
        width           = width,
        height          = height
    )

    mm.utils.plotting.save_video(
        frames,
        env.dt,
        Path(f'example/videos/{cfg.env_name}-rollout.mp4')
    )

    mm.utils.plotting.save_metrics(
        reward_plotter,
        Path(f'example/plots/{cfg.env_name}-reward.pdf')
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Roll out a policy and save a video and reward plot.')
    parser.add_argument('--config', type=Path, required=True, help='Path to the config file')
    parser.add_argument('--inference_fn', default='policy', choices=['policy', 'zero', 'random'],
                        help='Use the trained policy, or a dummy zero/random action function')
    parser.add_argument('--model_path', type=Path, default=None,
                        help='Checkpoint to load (defaults to the latest checkpoint in the run dir)')
    parser.add_argument('--T', type=float, default=10.0, help='Rollout duration in seconds')
    parser.add_argument('--width', type=int, default=640, help='Video width in pixels')
    parser.add_argument('--height', type=int, default=480, help='Video height in pixels')
    parser.add_argument('--camera', default='track', help='Camera name to render from')
    parser.add_argument('--stochastic', action='store_true', help='Sample actions instead of using the mean')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for the initial state')
    args = parser.parse_args()

    main(
        config_path       = args.config,
        inference_fn_type = args.inference_fn,
        model_path        = args.model_path,
        T                 = args.T,
        width             = args.width,
        height            = args.height,
        camera            = args.camera,
        deterministic     = not args.stochastic,
        seed              = args.seed
    )
