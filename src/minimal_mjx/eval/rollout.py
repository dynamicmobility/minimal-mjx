from minimal_mjx.utils.setupGPU import run_setup
from pathlib import Path
from minimal_mjx.learning.startup import read_config, create_environment, get_step_reset
from minimal_mjx.learning.inference import rollout, load_policy
from minimal_mjx.utils.plotting import save_video, save_metrics
import jax
import numpy as np
from mujoco_playground import wrapper

def main():
    # Set up the GPU environment
    # run_setup()
    
    # Read the config file from command line argument
    config = read_config()

    # Create the environment
    env, env_cfg = create_environment(config, idealistic=True, animate=False)
    
    # Load the model
    inference_fn = load_policy(config)
    
    # inference_fn = lambda obs, rng: (np.zeros(env.action_size, dtype=np.float32), None)
    jit_inference_fn = jax.jit(inference_fn)
    step, reset = get_step_reset(env, config['backend'])

    # Rollout the policy in the environment
    T = env.dt * 1000
    frames, reward_plotter, data_plotter, info_plotter = rollout(
        reset        = reset,
        step         = step, 
        inference_fn = jit_inference_fn,
        env          = env, 
        T            = T,
        width        = 640,
        height       = 480,
        show_progress= True
    )

    # Save metrics
    save_metrics(reward_plotter, path=Path(f'visualization/{config['env']}_metrics.pdf'))

    # Save the video
    save_video(
        frames,
        path=Path(f'visualization/{config["env"]}_rollout.mp4'),
        env_cfg=env_cfg,
    )
        

if __name__ == '__main__':
    main()