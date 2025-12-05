import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
from utils.setupGPU import run_setup
from pathlib import Path
from learning.training.startup import read_config, create_environment, get_step_reset, mo2so
from learning.inference import rollout, load_mo_policy, circle_vel, vx_sine_vel, load_so_policy
from utils.plotting import save_video, save_metrics, save_trajectories
import jax
import numpy as np
from mujoco_playground import wrapper

def set_disturbance(state):
    state.info['push_override'] = True
    state.info['push_override_xy'] = [30, 0.0]
    return state.info

def main():
    # Set up the GPU environment
    # run_setup()
    
    # Read the config file from command line argument
    config = read_config()

    # Create the environment
    env, env_cfg = create_environment(config, idealistic=True, animate=False)
    
    # Load the model
    directive = np.array([0.5, 0.5])
    if config['mo2so']['enabled']:
        print('Loading single objective policy')
        inference_fn = load_so_policy(config)
        # env = mo2so(env, weighting=[1.0, 0.0])
    else:
        print('Loading multi-objective policy')
        inference_fn = load_mo_policy(config, directive, deterministic=True)
    
    # inference_fn = lambda obs, rng: (np.zeros(env.action_size, dtype=np.float32), None)
    jit_inference_fn = jax.jit(inference_fn)
    reset, step = get_step_reset(env, config['backend'])

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
        # info_init_fn = set_disturbance
        # info_step_fn = lambda state: circle_vel(state, -vx_lim[1], vy_lim[1], 1 / T),
        # info_step_fn=lambda state: vx_sine_vel(state, 0.2, 0.2, 5 * 1 / T),
    )

    rewards = np.array(reward_plotter.rewards)
    # print(np.round(rewards[:20],2))
    # print(rewards.shape)
    print(np.sum(rewards, axis=0))
    print(directive)
    print(env.params.reward.optimization.objectives)

    # # Save metrics
    save_metrics(reward_plotter, path=Path(f'visualization/{config['env']}_metrics.pdf'))
    
    # # # Save trajectories
    # save_trajectories(
    #     env.mj_model,
    #     env.nq,
    #     data_plotter,
    #     pos_path=Path(f'visualization/{config["env"]}_position_trajectories.png'),
    #     vel_path=Path(f'visualization/{config["env"]}_velocity_trajectories.png'),
    #     torque_path=Path(f'visualization/{config["env"]}_torque_trajectories.png'),
    #     sensor_path=Path(f'visualization/{config["env"]}_sensor_trajectories.png'),
    # )

    # # Save the video
    save_video(
        frames,
        path=Path(f'visualization/{config["env"]}_rollout.mp4'),
        env_cfg=env_cfg,
    )
        

if __name__ == '__main__':
    main()