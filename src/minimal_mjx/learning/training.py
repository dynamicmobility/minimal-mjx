# Basic imports
from pathlib import Path
import time
import wandb
import datetime
import pickle
import pandas as pd
from ml_collections import config_dict
from datetime import datetime
from zoneinfo import ZoneInfo

# Graphics and plotting.
import matplotlib.pyplot as plt

# RL imports
import functools
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo.train import train as train_ppo
from brax.training.agents.ppo import networks as ppo_networks

# jax and MJX imports
from mujoco_playground import wrapper
import numpy as np

def setup_training(config):
    """Sets up the right training variables for the given algorithm"""
    ppo_params = config_dict.ConfigDict(config['ppo_params'])
    network_params = config_dict.ConfigDict(config['network_params'])
    return ppo_params, network_params


def train(
    config_yaml,
    output_dir: Path,
    env, 
    eval_env, 
    ppo_params, 
    network_params,
):
    train_algo = train_ppo
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **network_params
    )
    network_config = checkpoint.network_config(
        observation_size=eval_env.observation_size,
        action_size=eval_env.action_size,
        normalize_observations=ppo_params.normalize_observations,
        network_factory=network_factory,
    )
    run = initialize_wandb(name=f'{config_yaml['save_dir']}/{config_yaml['name']}') # move this outside...
    x_data, y_data, y_dataerr = [], [], []
    times = []
    train_fn = functools.partial(
        train_algo, **dict(ppo_params),
        network_factory=network_factory,
        progress_fn=lambda num_steps, metrics: plot_progress(
            num_steps  = num_steps,
            metrics    = metrics,
            times      = times,
            x_data     = x_data,
            y_data     = y_data,
            y_dataerr  = y_dataerr,
            ppo_params = ppo_params,
            save_dir   = output_dir,
            run        = run
        ),
        policy_params_fn=functools.partial(
            save_model,
            output_dir        = output_dir,
            run               = run,
            network_config    = network_config
        ),
    )
    
    make_inference_fn, trained_params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        eval_env=eval_env,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    
    return make_inference_fn, trained_params, metrics