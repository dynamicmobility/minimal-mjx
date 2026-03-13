import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import minimal_mjx as mm
from pathlib import Path
from example.create_environment import create_environment
import jax


config = mm.utils.read_config()
env, env_params = create_environment(
    config,
)
inference_fn = mm.eval.rollout.make_dummy_inference_fn(env, mode='zero')
inference_fn = jax.jit(inference_fn)


frames, reward_plotter, _, _ = mm.eval.rollout_policy(
    inference_fn    = inference_fn,
    env             = env,
    T               = 10.0,
    camera          = 'track',
    width           = 640,
    height          = 480
)

mm.utils.plotting.save_video(
    frames,
    env.dt,
    Path(f'example/videos/{config['env']}-simulate.mp4')
)

mm.utils.plotting.save_metrics(
    reward_plotter,
    Path(f'example/plots/{config['env']}-simulate.pdf')
)