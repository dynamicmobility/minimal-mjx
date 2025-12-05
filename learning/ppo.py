import os
os.environ["MUJOCO_GL"] = "egl"
# os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"
# os.environ['JAX_PLATFORMS']='cpu'
# os.environ['JAX_CHECK_TRACER_LEAKS']='true'

# Internal imports
from utils.setupGPU import run_setup
from learning.training.startup import read_config, create_environment, get_step_reset, get_commit_hash
from learning.training.training import setup_training, train
from learning.inference import rollout

# Basic imports
from pathlib import Path
import yaml
import datetime
import numpy as np

# Graphics and plotting.
import utils.plotting as plotting

# jax and MJX imports
import jax

"""
Repeat until converged:
  1. Collect N timesteps of interaction using current policy (possibly from many parallel envs).
  2. Compute returns and advantages (e.g., using GAE).
  3. Normalize advantages (optional but common).
  4. For K epochs:
       - Shuffle the collected data into minibatches
       - For each minibatch: compute policy loss (clipped surrogate), value loss, entropy bonus
       - Backprop and update policy + value networks (often shared or separate heads)
"""

def ppo_loss():
    pass

def minibatch_step():
    pass

def sgd_step():
    pass

def training_step():
    pass

def training_epoch():
    pass

def train():
    pass

def main():
    # Setup GPU
    run_setup()
    
    # Read the config file
    config = read_config()
    
    # Create the environment
    print('Creating environment...')
    env, env_cfg = create_environment(config, for_training=True)
    ppo_params, network_params = setup_training(config['learning_params'])
    
    # Train
    print('Training...')
    eval_env, env_cfg = create_environment(config, for_training=False)
    train()
    
if __name__ == "__main__":
    main()