import matplotlib
matplotlib.use('Agg')
import sys
import minimal_mjx as mm
from example.create_environment import create_environment

# Read configuration file
cfg = mm.utils.Config(sys.argv[1])

# Create environment
env, env_params = create_environment(cfg, for_training=True)

# Create the run directory and save the config next to the checkpoints
# (runs named 'test' may be overwritten and skip recording the git hash)
run_dir = mm.create_training_directory(cfg.run_dir, overwrite=cfg.is_test)
cfg.save(run_dir / 'config.yaml', make_dir=False)

# Initialize Weights and Biases Run object (if you want to use it)
USE_WANDB = True
if not USE_WANDB:
    run = None
else:
    run = mm.utils.logging.initialize_wandb(
        name    = str(run_dir).replace('/', ''),
        entity  = 'njanwani-gatech',
        project = 'my-awesome-project-2',
        config  = cfg.cfg.to_dict()
    )

# Begin training
mm.learning.train(
    env            = env,
    eval_env       = env,
    run_dir        = run_dir,
    ppo_params     = cfg.ppo_params,
    network_params = cfg.network_params,
    algorithm      = cfg.algorithm,
    run            = run,
    overwrite      = cfg.is_test,
)
