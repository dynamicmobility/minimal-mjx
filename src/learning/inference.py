# Basic imports
from pathlib import Path
from etils import epath
import numpy as np
from glob import glob
from tqdm import tqdm

# Internal imports
import utils.plotting as plotting
from utils.state import MujocoState

# RL imports
from brax.training.agents.ppo import checkpoint

# jax and MJX imports
from mujoco_playground._src.mjx_env import MjxEnv
import jax

def get_all_models(config: dict, sort=True) -> Path:
    """Returns the last model file in the model directory."""
    model_dir = Path(config['save_dir']) / config['name']
    dir_files = glob(str(model_dir / '*'))
    model_files = [Path(f) for f in dir_files if '.' not in f]
    if sort:
        model_files.sort(key=lambda x: int(x.name))
    return model_files

def get_last_model(config: dict) -> Path:
    """Returns the last model file in the model directory."""
    model_files = get_all_models(config, sort=True)
    
    return model_files[-1]

def load_policy(config, deterministic=True):
    path = get_last_model(config)
    print(f'Loading model at {path.as_posix()}')
    policy = checkpoint.load_policy(
        path.resolve(),
        deterministic=deterministic
    )
    return policy

def get_params(
    config: dict,
    path: Path = None,
    silent: bool = False
):
    if path is None:
        path = get_last_model(config)
    if not silent:
        print(f'Loading model at {path.as_posix()}')
    fullpath = path.resolve()
    
    fullpath = epath.Path(fullpath)
    params = checkpoint.load(fullpath)
    return params
    
def load_policy(config, deterministic=True):
    path = get_last_model(config)
    print(f'Loading model at {path.as_posix()}')
    policy = checkpoint.load_policy(
        path.resolve(),
        deterministic=deterministic
    )
    return policy

def infer_frame_dim(
    mj_model, width, height
):
    if width is None:
        width = mj_model.vis.global_.offwidth
    if height is None:
        height = mj_model.vis.global_.offheight
    
    return width, height


def rollout(
    reset,
    step,
    inference_fn,
    env           : MjxEnv,
    T             = 10.0,
    info_init_fn  = lambda state: state.info,
    info_step_fn  = lambda state: state.info,
    info_plot_key = None,
    width         = None,
    height        = None,
    gen_vid       = True,
    show_progress = True
) -> tuple[list, plotting.RewardPlotter, plotting.MujocoPlotter, plotting.InfoPlotter]:
    width, height = infer_frame_dim(env.mj_model, width, height)
    
    # Set up the environment
    rng = jax.random.PRNGKey(np.random.randint(0, 100000))
    state: MujocoState = reset(rng)
    
    # Initialize the state
    initial_info = state.info | info_init_fn(state)
    state = state.replace(info=initial_info)

    # Setup reward plotting
    reward_plotter = plotting.RewardPlotter(state.metrics)
    data_plotter = plotting.MujocoPlotter()
    info_plotter = plotting.InfoPlotter(plotkey=info_plot_key)
    data_plotter.add_row(state.data)

    # Rollout and record data
    N = int(T / env.dt)
    traj = [state]
    for i in tqdm(range(N), disable=not show_progress):
        ctrl, _ = inference_fn(state.obs, rng)
        state = step(state, ctrl)
        state = state.replace(info=state.info | info_step_fn(state))
        data_plotter.add_row(state.data)
        reward_plotter.add_row(state.metrics, state.reward)
        info_plotter.add_row(state.data.time, state.info)
        traj.append(state)
        
        if state.done:
            break

    scene_option = plotting.get_mj_scene_option(contacts=False, com=False)

    if gen_vid:
        print('Generating video...')
        frames = env.render(
            trajectory   = traj,
            camera       = 'track',
            height       = height,
            width        = width,
            scene_option = scene_option,
        )
    else:
        frames = None
    return frames, reward_plotter, data_plotter, info_plotter