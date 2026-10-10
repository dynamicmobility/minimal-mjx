import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import sys
import minimal_mjx as mm
from pathlib import Path
from example.create_environment import create_environment
import jax


cfg = mm.utils.Config(sys.argv[1])
env, env_params = create_environment(cfg)

inference_fn = mm.learning.inference.load_policy(cfg.run_dir, deterministic=True)
inference_fn = jax.jit(inference_fn)

frames, _, reward_plotter, _, _ = mm.eval.rollout_policy(
    inference_fn    = inference_fn,
    env             = env,
    n_steps         = int(10.0 / env.dt),
    camera          = 'track',
    width           = 640,
    height          = 480
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