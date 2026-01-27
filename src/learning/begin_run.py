import os

# Internal imports
from utils.setupGPU import run_setup
from learning.startup import read_config, create_environment, get_commit_hash
from learning.training import setup_training, train

# Basic imports
from pathlib import Path
import yaml

def main():
    run_setup()
    
    # Read the config file
    config = read_config()
    output_dir = Path(config['save_dir']) / config['name']
    os.makedirs(output_dir, exist_ok=config['name'] == 'test')
    
    # Create the environment
    print('Creating environment...')
    env, env_cfg = create_environment(config, for_training=True)
    ppo_params, network_params = setup_training(config.learning_params)

    # Save configuration
    config_save_path = Path(output_dir) / 'config.yaml'
    if config.name != 'test':
        git_hash = get_commit_hash()
        config.git_hash = git_hash
    with open(config_save_path, 'w') as f:
        yaml.dump(config.to_dict(), f)
    
    # Data for plotting
    print('Setting up data for plotting...')
    
    # Train
    print('Training...')
    eval_env, _ = create_environment(config, for_training=False)
    _ = train(
        config, output_dir, env, eval_env, ppo_params, network_params
    )
    
    
if __name__ == "__main__":
    main()
