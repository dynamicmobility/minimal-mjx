import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
# os.environ['JAX_TRACEBACK_FILTERING']='off'

from utils.setupGPU import run_setup
run_setup() # Run the GPU setup

# Python imports
import argparse
import numpy as np
from pathlib import Path

# Local imports
from learning.startup import create_environment, get_step_reset, read_config
from learning.inference import rollout
from utils.plotting import save_video, save_metrics
    
def main():
    # Load arguments
    config = read_config()
    env_name = config['env']
    
    # Loading the environment
    env, env_cfg = create_environment(config, idealistic=False, animate=False)

    # Get reset and step functions
    reset, step = get_step_reset(env, config['backend'])
    
    # Simulate the environment
    def inference_fn(obs, rng):
        """Dummy inference function for the environment."""
        ctrl = np.zeros(env.action_size)
        # ctrl = np.random.random(env.action_size) * 2 - 1
        # ctrl = np.ones(env.action_size) * 1.5  
        return ctrl, None

    T = env.dt * 500
    frames, reward_plotter, data_plotter, info_plotter= rollout(
        reset,
        step,
        inference_fn,
        env,
        T=T,
        height=640,
        width=480,
    )

    save_metrics(reward_plotter, path=Path(f'visualization/{env_name}_metrics.png'))
    save_video(frames, env_cfg, path=Path(f'visualization/{env_name}_sim.mp4'))
    
if __name__ == '__main__':
    main()